"""领域服务：上传入库、锚点维护、查询、状态汇总。

所有写操作在单事务内完成并递增 state_generation（触发后续重算）。
"""
from __future__ import annotations

from datetime import datetime

from .config import Settings
from .db import Database
from .timeutil import fmt_utc, utcnow_iso


class ServiceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        code: str = "BAD_REQUEST",
        errors: list | None = None,
        current_version: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.errors = errors or []
        self.current_version = current_version


def _bump_generation(cur) -> None:
    cur.execute(
        "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        " WHERE key = 'state_generation'"
    )


# ---------------------------------------------------------------- 上传

def add_batch(db: Database, cfg: Settings, filename: str, rows: list[dict]) -> int:
    """整批入库（原子）：任一点冲突或超上限则整批拒绝。返回批次号。"""
    if not rows:
        raise ServiceError("没有可入库的数据行", code="EMPTY_BATCH")
    with db.tx() as cur:
        total = cur.execute("SELECT COUNT(*) AS c FROM readings").fetchone()["c"]
        if total + len(rows) > cfg.max_points:
            raise ServiceError(
                f"累计点数将超过上限 {cfg.max_points}（现有 {total} 点，本批 {len(rows)} 行），整批已拒绝",
                code="POINT_LIMIT_EXCEEDED",
            )
        conflicts = []
        for r in rows:
            hit = cur.execute(
                "SELECT 1 FROM readings WHERE segment = ? AND local_seconds = ?",
                (r["segment"], r["local_seconds"]),
            ).fetchone()
            if hit:
                conflicts.append({
                    "row": r["row_no"],
                    "message": f"与已有数据重复：段 {r['segment']} 秒数 {r['local_seconds']}",
                })
        if conflicts:
            raise ServiceError(
                f"发现 {len(conflicts)} 条同段秒数重复，整批已拒绝",
                code="DUPLICATE_SECONDS",
                errors=conflicts,
            )
        now = utcnow_iso()
        batch_id = cur.execute(
            "INSERT INTO batches(filename, created_at, rows_accepted) VALUES (?, ?, ?)",
            (filename[:200], now, len(rows)),
        ).lastrowid
        cur.executemany(
            "INSERT INTO readings(batch_id, row_no, segment, local_seconds, temperature, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (batch_id, r["row_no"], r["segment"], r["local_seconds"], r["temperature"], now)
                for r in rows
            ],
        )
        _bump_generation(cur)
    return batch_id


# ---------------------------------------------------------------- 锚点

def set_anchors(
    db: Database,
    segment: int,
    local1: int,
    utc1,                      # datetime
    local2: int,
    utc2,                      # datetime
    version: int | None,       # 所见版本；新建时须为 None
) -> int:
    """录入或更正某段的两枚锚点，返回新版本号。

    约束：0 ≤ local1 < local2 且 utc1 < utc2（UTC 须递增）。
    更正时必须提交所见版本，否则 409。
    """
    if segment < 0:
        raise ServiceError("段号须为非负整数", code="BAD_SEGMENT")
    if not (0 <= local1 < local2):
        raise ServiceError("本地秒数须满足 0 ≤ 秒数1 < 秒数2", code="BAD_LOCAL_RANGE")
    if not (utc1 < utc2):
        raise ServiceError("UTC 必须递增：UTC2 须晚于 UTC1", code="UTC_NOT_INCREASING")

    with db.tx() as cur:
        row = cur.execute(
            "SELECT version FROM anchors WHERE segment = ?", (segment,)
        ).fetchone()
        now = utcnow_iso()
        if row is None:
            if version is not None:
                raise ServiceError(
                    "该段尚无锚点，无法按版本更正（请清空版本字段按新建提交）",
                    status=409, code="VERSION_CONFLICT", current_version=None,
                )
            new_version = 1
            cur.execute(
                "INSERT INTO anchors(segment, local1, utc1, local2, utc2, version, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (segment, local1, fmt_utc(utc1), local2, fmt_utc(utc2), new_version, now),
            )
        else:
            current = row["version"]
            if version is None or version != current:
                raise ServiceError(
                    f"锚点版本不一致：提交版本 {version}，当前版本 {current}，请刷新后重试",
                    status=409, code="VERSION_CONFLICT", current_version=current,
                )
            new_version = current + 1
            cur.execute(
                "UPDATE anchors SET local1 = ?, utc1 = ?, local2 = ?, utc2 = ?,"
                " version = ?, updated_at = ? WHERE segment = ?",
                (local1, fmt_utc(utc1), local2, fmt_utc(utc2), new_version, now, segment),
            )
        _bump_generation(cur)
    return new_version


# ---------------------------------------------------------------- 查询
#
# 供数归属原则：一次查询的版号、日期、读数与分类必须来自同一已发布结果，
# 因此每个查询函数都在单个 db.snapshot() 内取齐全部数据（锁内串行，换版
# 提交不会插入其中）。读数按三态归属：
#   已发布 —— 在当前 calibrated 版本中，按日结果可读；绝不再列入待办；
#   待发布 —— 未入版但已被当前锚点覆盖，下次换版即转入按日结果；
#   待校时 —— 无锚点或超出锚点区间，需先补/改锚点，附原因仍可定位。

def _calc_info_conn(conn) -> dict:
    row = conn.execute("SELECT * FROM calc_state WHERE id = 1").fetchone()
    meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    state_gen = int(meta.get("state_generation", "0"))
    calibrated_gen = int(meta.get("calibrated_generation", "-1"))
    return {
        "version": row["version"],
        "status": row["status"],
        "last_error": row["last_error"],
        "updated_at": row["updated_at"],
        "state_generation": state_gen,
        "calibrated_generation": calibrated_gen,
        "dirty": state_gen != calibrated_gen,
    }


def calc_info(db: Database) -> dict:
    return db.snapshot(_calc_info_conn)


def _list_anchors_conn(conn) -> list[dict]:
    anchors = [dict(a) for a in conn.execute("SELECT * FROM anchors ORDER BY segment")]
    secs_by_seg: dict[int, list[int]] = {}
    for r in conn.execute("SELECT segment, local_seconds FROM readings"):
        secs_by_seg.setdefault(r["segment"], []).append(r["local_seconds"])
    for a in anchors:
        secs = secs_by_seg.get(a["segment"], [])
        a["point_count"] = len(secs)
        a["in_range_count"] = sum(1 for s in secs if a["local1"] <= s <= a["local2"])
    return anchors


def list_anchors(db: Database) -> list[dict]:
    return db.snapshot(_list_anchors_conn)


def _classify_conn(conn) -> tuple[list[dict], list[dict]]:
    """把未入版读数分为（待发布, 待校时）；已发布的一律不列入待办。

    判定基于同一快照内的已发布集合与当前锚点：换版提交不会插入其中，
    因此分类与同时返回的按日结果必然同属一个已发布版本。
    """
    anchors = {a["segment"]: a for a in conn.execute("SELECT * FROM anchors")}
    published = {
        r["reading_id"] for r in conn.execute("SELECT reading_id FROM calibrated")
    }
    pending_publish: list[dict] = []
    uncalibratable: list[dict] = []
    for r in conn.execute(
        "SELECT id, batch_id, row_no, segment, local_seconds, temperature"
        " FROM readings ORDER BY id"
    ):
        d = dict(r)
        if d["id"] in published:
            continue  # 已发布：旧版仍可读，不得误报待校时
        a = anchors.get(d["segment"])
        if a is None:
            uncalibratable.append({**d, "reason": "该段未设置校时锚点"})
        elif not (a["local1"] <= d["local_seconds"] <= a["local2"]):
            uncalibratable.append(
                {**d, "reason": f"超出锚点范围 [{a['local1']}, {a['local2']}]"}
            )
        else:
            pending_publish.append({
                **d,
                "anchor_version": a["version"],
                "reason": "已受当前锚点覆盖，等待重算发布入版",
            })
    return pending_publish, uncalibratable


_DAY_SQL = """
SELECT c.reading_id, c.utc_time, c.anchor_version, c.calc_version,
       r.batch_id, r.row_no, r.segment, r.local_seconds, r.temperature
FROM calibrated c JOIN readings r ON r.id = c.reading_id
WHERE c.utc_date = ?
ORDER BY c.utc_time, c.reading_id
"""


def _parse_day(date_str: str):
    try:
        return datetime.strptime((date_str or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ServiceError("日期格式须为 YYYY-MM-DD", code="BAD_DATE")


def _day_result_conn(conn, day) -> dict:
    """同一快照内取 版号 + 当日读数：响应版号与每行读数版号必然一致。"""
    info = _calc_info_conn(conn)
    rows = conn.execute(_DAY_SQL, (day.isoformat(),)).fetchall()
    return {
        "date": day.isoformat(),
        "calc_version": info["version"],
        "dirty": info["dirty"],
        "count": len(rows),
        "readings": [dict(r) for r in rows],
    }


def query_by_date(db: Database, date_str: str) -> dict:
    """按 UTC 日查询已发布读数；版号、日期、读数来自同一已发布结果。"""
    day = _parse_day(date_str)
    return db.snapshot(lambda conn: _day_result_conn(conn, day))


def query_page_data(db: Database, date_str: str) -> dict:
    """查询页整页数据：按日结果、待发布/待校时分类、重算状态，同一快照。"""
    day = _parse_day(date_str) if (date_str or "").strip() else None

    def fn(conn):
        pending_publish, uncalibratable = _classify_conn(conn)
        return {
            "result": _day_result_conn(conn, day) if day is not None else None,
            "calc": _calc_info_conn(conn),
            "pending_publish": pending_publish,
            "pending": uncalibratable,
        }

    return db.snapshot(fn)


def pending_overview(db: Database) -> dict:
    """待发布 + 待校时总览（同一快照，附所基于的已发布版号）。"""
    def fn(conn):
        info = _calc_info_conn(conn)
        pending_publish, uncalibratable = _classify_conn(conn)
        return {
            "calc_version": info["version"],
            "dirty": info["dirty"],
            "count": len(pending_publish) + len(uncalibratable),
            "pending_publish_count": len(pending_publish),
            "pending_count": len(uncalibratable),
            "pending_publish": pending_publish,
            "pending": uncalibratable,
        }

    return db.snapshot(fn)


def get_status(db: Database, cfg: Settings) -> dict:
    """状态页/接口汇总：点数、重算状态、锚点、批次、两类待办，同一快照。"""
    def fn(conn):
        accepted = conn.execute("SELECT COUNT(*) AS c FROM readings").fetchone()["c"]
        batches = [
            dict(b)
            for b in conn.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 100")
        ]
        pending_publish, uncalibratable = _classify_conn(conn)
        return {
            "points": {"accepted": accepted, "max": cfg.max_points},
            "calc": _calc_info_conn(conn),
            "anchors": _list_anchors_conn(conn),
            "batches": batches,
            "pending_count": len(uncalibratable),
            "pending_publish_count": len(pending_publish),
        }

    return db.snapshot(fn)
