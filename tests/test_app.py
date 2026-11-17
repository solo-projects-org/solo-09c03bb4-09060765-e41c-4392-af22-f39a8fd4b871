"""关键流程自测：上传、锚点、校时、跨日原子性、并发交错、重启恢复、失败不留半版、
供数归属（已发布/待发布/待校时 三态与版号一致性）。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

CSV_HEADER = "segment,local_seconds,temperature\n"


def make_client(tmp_path, monkeypatch, worker="0", delay="0", max_points=None) -> TestClient:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("WORKER_ENABLED", worker)
    monkeypatch.setenv("RECALC_DELAY", delay)
    if max_points is not None:
        monkeypatch.setenv("MAX_POINTS", str(max_points))
    from app.main import create_app
    return TestClient(create_app())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as c:
        yield c


def upload(client: TestClient, rows, header: str | None = CSV_HEADER, filename="data.csv"):
    content = (header or "") + "".join(f"{s},{sec},{t}\n" for s, sec, t in rows)
    return client.post("/api/uploads", files={"file": (filename, content, "text/csv")})


def set_anchors(client: TestClient, seg, l1, u1, l2, u2, version=None):
    payload = {"segment": seg, "local1": l1, "utc1": u1, "local2": l2, "utc2": u2}
    if version is not None:
        payload["version"] = version
    return client.post("/api/anchors", json=payload)


def wait_idle(client: TestClient, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        calc = client.get("/api/status").json()["calc"]
        if calc["status"] != "running" and not calc["dirty"]:
            return calc
        time.sleep(0.05)
    raise AssertionError("重算未在超时内收敛")


# ---------------------------------------------------------------- 上传

def test_upload_happy(client):
    r = upload(client, [(1, 100, 4.5), (1, 101, 4.6), (2, 5, -2.0)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == 3 and body["batch_id"] == 1
    st = client.get("/api/status").json()
    assert st["points"] == {"accepted": 3, "max": 2000}
    assert st["batches"][0]["rows_accepted"] == 3


def test_duplicate_within_file_rejected(client):
    r = upload(client, [(1, 100, 1.0), (1, 100, 2.0)])
    assert r.status_code == 400
    assert r.json()["code"] == "CSV_INVALID"
    assert any("重复" in e["message"] for e in r.json()["errors"])
    assert client.get("/api/status").json()["points"]["accepted"] == 0


def test_duplicate_against_db_rejected(client):
    assert upload(client, [(1, 100, 1.0)]).status_code == 200
    r = upload(client, [(1, 100, 9.9)])
    assert r.status_code == 400 and r.json()["code"] == "DUPLICATE_SECONDS"
    # 不同段同秒数允许
    assert upload(client, [(2, 100, 3.0)]).status_code == 200
    assert client.get("/api/status").json()["points"]["accepted"] == 2


def test_point_cap_enforced(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, max_points="5") as c:
        assert upload(c, [(1, i, 1.0) for i in range(4)]).status_code == 200
        r = upload(c, [(1, 10, 1.0), (1, 11, 1.0)])
        assert r.status_code == 400 and r.json()["code"] == "POINT_LIMIT_EXCEEDED"
        assert c.get("/api/status").json()["points"]["accepted"] == 4
        assert upload(c, [(1, 10, 1.0)]).status_code == 200
        assert c.get("/api/status").json()["points"]["accepted"] == 5


def test_bad_rows_reported_with_row_numbers(client):
    content = "segment,local_seconds,temperature\n1,100,abc\n2,x,1.0\n"
    r = client.post("/api/uploads", files={"file": ("b.csv", content, "text/csv")})
    assert r.status_code == 400
    assert {e["row"] for e in r.json()["errors"]} == {1, 2}
    assert client.get("/api/status").json()["points"]["accepted"] == 0


def test_csv_chinese_header_bom_and_no_header(client):
    content = "﻿开机段号,本地秒数,温度\n1,100,4.5\n"
    r = client.post("/api/uploads", files={"file": ("c.csv", content.encode("utf-8"), "text/csv")})
    assert r.status_code == 200, r.text
    r = client.post("/api/uploads",
                    files={"file": ("d.csv", "2,100,4.5\n2,101,4.6\n", "text/csv")})
    assert r.status_code == 200 and r.json()["accepted"] == 2


# ---------------------------------------------------------------- 锚点

def test_anchor_create_and_validation(client):
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    assert r.status_code == 200 and r.json()["version"] == 1
    # UTC 不递增
    r = set_anchors(client, 2, 0, "2026-01-02T00:00:00Z", 10, "2026-01-01T00:00:00Z")
    assert r.status_code == 400 and r.json()["code"] == "UTC_NOT_INCREASING"
    # 本地秒数不递增
    r = set_anchors(client, 2, 10, "2026-01-01T00:00:00Z", 10, "2026-01-02T00:00:00Z")
    assert r.status_code == 400 and r.json()["code"] == "BAD_LOCAL_RANGE"


def test_anchor_correction_requires_seen_version(client):
    assert set_anchors(client, 1, 100, "2026-01-01T00:00:00Z",
                     200, "2026-01-01T00:10:00Z").json()["version"] == 1
    # 未带版本
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z")
    assert r.status_code == 409 and r.json()["current_version"] == 1
    # 版本不符
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z",
                    version=3)
    assert r.status_code == 409 and r.json()["current_version"] == 1
    # 版本正确
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z",
                    version=1)
    assert r.status_code == 200 and r.json()["version"] == 2
    # 对不存在的段按版本更正
    r = set_anchors(client, 9, 0, "2026-01-01T00:00:00Z", 10, "2026-01-01T00:10:00Z",
                    version=1)
    assert r.status_code == 409 and r.json()["current_version"] is None


# ---------------------------------------------------------------- 校时

def test_calibration_math_and_pending(client):
    upload(client, [(1, 99, 1.0), (1, 100, 2.0), (1, 150, 3.0),
                    (1, 200, 4.0), (1, 201, 5.0), (2, 100, 6.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert body["count"] == 3
    times = {row["local_seconds"]: row["utc_time"] for row in body["readings"]}
    assert times[100] == "2026-01-01T00:00:00.000Z"
    assert times[150] == "2026-01-01T00:05:00.000Z"   # 线性中点
    assert times[200] == "2026-01-01T00:10:00.000Z"
    # 查询结果须含批次号、原行号、锚点版本
    for row in body["readings"]:
        assert row["batch_id"] == 1
        assert row["anchor_version"] == 1
        assert row["row_no"] >= 1
    rows_by_sec = {row["local_seconds"]: row["row_no"] for row in body["readings"]}
    assert rows_by_sec == {100: 2, 150: 3, 200: 4}
    # 范围外与无锚点的读数留存待校时
    pend = client.get("/api/pending").json()
    got = sorted((p["segment"], p["local_seconds"], p["reason"]) for p in pend["pending"])
    assert [(s, sec) for s, sec, _ in got] == [(1, 99), (1, 201), (2, 100)]
    assert "未设置" in got[2][2] and "超出" in got[0][2]


def test_day_crossing_moves_atomically(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0)])
    # 100 → 1日23:00，150 → 2日00:00（恰跨日）
    set_anchors(client, 1, 100, "2026-01-01T23:00:00Z", 200, "2026-01-02T01:00:00Z")
    d1 = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    d2 = client.get("/api/readings", params={"date": "2026-01-02"}).json()
    assert d1["count"] == 1 and d2["count"] == 1
    # 更正锚点，整体后移一天：旧日移除与新日加入一起生效
    set_anchors(client, 1, 100, "2026-01-02T23:00:00Z", 200, "2026-01-03T01:00:00Z",
                version=1)
    d1 = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    d2 = client.get("/api/readings", params={"date": "2026-01-02"}).json()
    d3 = client.get("/api/readings", params={"date": "2026-01-03"}).json()
    assert d1["count"] == 0 and d2["count"] == 1 and d3["count"] == 1
    assert d2["readings"][0]["anchor_version"] == 2
    assert d2["calc_version"] == d3["calc_version"]  # 同一次换版


# ---------------------------------------------------------------- 并发 / 恢复

def test_interleaved_upload_during_recalc(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        v1 = c.get("/api/status").json()["calc"]["version"]

        upload(c, [(1, 150, 2.0)])  # 触发一次带延迟的重算
        t0 = time.time()
        while time.time() - t0 < 5:  # 等它进入重算中
            if c.get("/api/status").json()["calc"]["status"] == "running":
                break
            time.sleep(0.02)
        r = upload(c, [(1, 160, 3.0)])  # 重算期间上传新点，须被接纳
        assert r.status_code == 200

        st = wait_idle(c)
        assert st["version"] > v1  # 版本单调递增，不回退
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(row["local_seconds"] for row in body["readings"]) == [100, 150, 160]


def test_old_version_visible_during_recalc(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        # 更正锚点使读数跨到次日；重算带延迟，期间旧版仍完整可见
        set_anchors(c, 1, 100, "2026-01-02T00:00:00Z", 200, "2026-01-02T00:10:00Z",
                    version=1)
        d1 = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        d2 = c.get("/api/readings", params={"date": "2026-01-02"}).json()
        assert d1["count"] == 1 and d2["count"] == 0
        wait_idle(c)
        d1 = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        d2 = c.get("/api/readings", params={"date": "2026-01-02"}).json()
        assert d1["count"] == 0 and d2["count"] == 1
        assert d2["readings"][0]["anchor_version"] == 2


def test_restart_recovers_pending_recalc(tmp_path, monkeypatch):
    from app import calibration
    from app.main import create_app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("WORKER_ENABLED", "0")
    # 模拟“崩溃前没来得及重算”：屏蔽同步重算，只落库
    orig = calibration.RecalcWorker.run_until_quiet
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", lambda self: None)
    with TestClient(create_app()) as c1:
        upload(c1, [(1, 100, 1.0)])
        set_anchors(c1, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        assert c1.get("/api/status").json()["calc"]["dirty"] is True
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", orig)
    # 重启后：后台线程自动补算，已接纳且在锚点范围内的点最终进入当前版
    monkeypatch.setenv("WORKER_ENABLED", "1")
    with TestClient(create_app()) as c2:
        wait_idle(c2)
        body = c2.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert body["count"] == 1
        assert body["readings"][0]["local_seconds"] == 100


def test_failed_recalc_keeps_old_version(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as c:  # 同步重算模式
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        before = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert before["count"] == 1
        v_before = c.get("/api/status").json()["calc"]["version"]

        from app import calibration

        def boom(readings, anchors):
            raise RuntimeError("模拟计算失败")

        monkeypatch.setattr(calibration, "compute_calibration", boom)
        r = upload(c, [(1, 150, 2.0)])  # 新点仍被接纳，但重算失败
        assert r.status_code == 200
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "failed"
        assert calc["version"] == v_before          # 版本没有推进
        after = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert after["count"] == 1                   # 旧版完整，无半版
        assert [row["local_seconds"] for row in after["readings"]] == [100]
        # 失败期间：旧版已发布点不得误报待校时；新点受锚点覆盖但未入版 → 待发布
        pend = c.get("/api/pending").json()
        assert pend["pending"] == []
        assert [p["local_seconds"] for p in pend["pending_publish"]] == [150]
        assert pend["pending_publish"][0]["anchor_version"] == 1

        monkeypatch.undo()  # 恢复 compute_calibration
        c.post("/api/recalc")
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "idle" and not calc["dirty"]
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(row["local_seconds"] for row in body["readings"]) == [100, 150]
        pend = c.get("/api/pending").json()
        assert pend["pending"] == [] and pend["pending_publish"] == []


# ---------------------------------------------------------------- 供数归属

def test_published_not_pending_while_old_version_readable(tmp_path, monkeypatch):
    """锚点更正收窄区间：旧版仍可读时，已发布点不得误报待校时；换版后分类一起更新。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.8") as c:
        upload(c, [(1, 100, 1.0), (1, 150, 2.0), (1, 200, 3.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        # 更正锚点收窄到 [100,150]：点 200 在新区间外，但旧版仍发布它
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 150, "2026-01-01T00:05:00Z",
                    version=1)
        # 重算延迟窗口内：按日结果仍是旧版三点，且三点都不在任何待办列表中
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(r["local_seconds"] for r in d["readings"]) == [100, 150, 200]
        pend = c.get("/api/pending").json()
        assert pend["pending"] == []
        assert pend["pending_publish"] == []
        wait_idle(c)
        # 新版本生效：按日结果与分类一起更新 —— 200 退出已发布，转为待校时
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(r["local_seconds"] for r in d["readings"]) == [100, 150]
        pend = c.get("/api/pending").json()
        assert [(p["segment"], p["local_seconds"]) for p in pend["pending"]] == [(1, 200)]
        assert "超出" in pend["pending"][0]["reason"]
        assert pend["pending_publish"] == []


def test_new_point_pending_publish_until_committed(tmp_path, monkeypatch):
    """重算期间上传：新点未入版但受锚点覆盖 → 待发布；换版后转入按日结果。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.8") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        v1 = c.get("/api/status").json()["calc"]["version"]

        upload(c, [(1, 150, 2.0)])  # 触发带延迟的重算
        # 窗口内：旧版按日结果不变；新点列为待发布而非待校时
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert [r["local_seconds"] for r in d["readings"]] == [100]
        assert d["calc_version"] == v1
        pend = c.get("/api/pending").json()
        assert [p["local_seconds"] for p in pend["pending_publish"]] == [150]
        assert pend["pending_publish"][0]["batch_id"] == 2
        assert pend["pending_publish"][0]["row_no"] == 1
        assert pend["pending"] == []
        wait_idle(c)
        # 新版本生效：新点入版，待发布清空，批次号/原行号/锚点版本保留
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert d["calc_version"] > v1
        row = next(r for r in d["readings"] if r["local_seconds"] == 150)
        assert row["batch_id"] == 2 and row["row_no"] == 1 and row["anchor_version"] == 1
        pend = c.get("/api/pending").json()
        assert pend["pending_publish"] == [] and pend["pending"] == []


def test_response_version_always_matches_rows(tmp_path, monkeypatch):
    """换版恰逢查询：响应版号、日期与每行读数版号始终同属一个已发布版本。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.3") as c:
        upload(c, [(1, 100, 1.0), (1, 150, 2.0)])
        set_anchors(c, 1, 100, "2026-01-01T23:00:00Z", 200, "2026-01-02T01:00:00Z")
        wait_idle(c)
        # 跨日更正 + 连续触发多次换版，期间反复查询两个日期
        set_anchors(c, 1, 100, "2026-01-02T23:00:00Z", 200, "2026-01-03T01:00:00Z",
                    version=1)
        upload(c, [(1, 120, 3.0)])
        for _ in range(40):
            for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
                d = c.get("/api/readings", params={"date": day}).json()
                assert d["date"] == day
                assert all(r["calc_version"] == d["calc_version"] for r in d["readings"])
                pend = c.get("/api/pending").json()
                # 任何读数不得同时出现在按日结果与待办列表中
                published_secs = {(r["segment"], r["local_seconds"]) for r in d["readings"]}
                for lst in (pend["pending"], pend["pending_publish"]):
                    todo_secs = {(q["segment"], q["local_seconds"]) for q in lst}
                    assert not todo_secs & published_secs
            time.sleep(0.02)
        wait_idle(c)
        # 三点最终全部入版：100/120 → 01-02，150 → 01-03（恰跨日）
        d2 = c.get("/api/readings", params={"date": "2026-01-02"}).json()
        d3 = c.get("/api/readings", params={"date": "2026-01-03"}).json()
        assert d2["count"] == 2 and d3["count"] == 1
        assert d2["calc_version"] == d3["calc_version"]  # 同属最后一次换版


def test_failed_recalc_after_correction_keeps_attribution(tmp_path, monkeypatch):
    """更正锚点 + 重算失败：旧版已发布点不误报待校时；恢复后分类随新版一起更新。"""
    with make_client(tmp_path, monkeypatch) as c:  # 同步重算模式
        upload(c, [(1, 100, 1.0), (1, 200, 2.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        v1 = c.get("/api/status").json()["calc"]["version"]

        from app import calibration

        def boom(readings, anchors):
            raise RuntimeError("模拟计算失败")

        monkeypatch.setattr(calibration, "compute_calibration", boom)
        # 更正锚点收窄到 [100,150]：点 200 将超出新区间，但重算失败 → 旧版保留
        r = set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 150, "2026-01-01T00:05:00Z",
                        version=1)
        assert r.status_code == 200
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "failed" and calc["version"] == v1
        # 旧版两点仍完整可读，且都不列入任何待办
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(x["local_seconds"] for x in d["readings"]) == [100, 200]
        assert d["calc_version"] == v1
        pend = c.get("/api/pending").json()
        assert pend["pending"] == [] and pend["pending_publish"] == []

        monkeypatch.undo()
        c.post("/api/recalc")
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "idle" and not calc["dirty"]
        # 新版生效：200 退出按日结果并转为待校时（超出新区间）
        d = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert [x["local_seconds"] for x in d["readings"]] == [100]
        pend = c.get("/api/pending").json()
        assert [(p["segment"], p["local_seconds"]) for p in pend["pending"]] == [(1, 200)]


def test_restart_keeps_pending_publish_then_recovers(tmp_path, monkeypatch):
    """重启后：未入版但受锚点覆盖的点仍标为待发布，补算后转入按日结果。"""
    from app import calibration
    from app.main import create_app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("WORKER_ENABLED", "0")
    # 模拟“崩溃前没来得及重算”：屏蔽同步重算，只落库
    orig = calibration.RecalcWorker.run_until_quiet
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", lambda self: None)
    with TestClient(create_app()) as c1:
        upload(c1, [(1, 100, 1.0)])
        set_anchors(c1, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        pend = c1.get("/api/pending").json()
        assert [p["local_seconds"] for p in pend["pending_publish"]] == [100]
        assert pend["pending"] == []
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", orig)
    # 重启（后台线程仍关闭）：分类关系保持 —— 待发布不丢、不错标为待校时
    with TestClient(create_app()) as c2:
        pend = c2.get("/api/pending").json()
        assert [p["local_seconds"] for p in pend["pending_publish"]] == [100]
        assert pend["pending"] == []
        assert c2.get("/api/readings", params={"date": "2026-01-01"}).json()["count"] == 0
        c2.post("/api/recalc")  # 手动补算
        d = c2.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert d["count"] == 1 and d["readings"][0]["local_seconds"] == 100
        pend = c2.get("/api/pending").json()
        assert pend["pending_publish"] == [] and pend["pending"] == []


# ---------------------------------------------------------------- 页面

def test_html_pages_render(client):
    upload(client, [(1, 100, 1.0), (2, 7, 2.0)])  # 段 2 无锚点 → 待校时
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    for path in ["/", "/upload", "/anchors", "/query?date=2026-01-01"]:
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path
    html = client.get("/query?date=2026-01-01").text
    assert "批次号" in html and "原行号" in html and "锚点版本" in html
    assert "待发布读数" in html and "待校时读数" in html
    assert "未设置" in html  # 段 2 的待校时原因
    assert "待发布" in client.get("/").text  # 状态页含待发布计数
