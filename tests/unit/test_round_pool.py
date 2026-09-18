from __future__ import annotations

import threading
import time

from urban_intervention.causal.label_queue.round_pool import (
    RoundPool,
    RunIdentity,
)


def identity(run_id: str = "run-test") -> RunIdentity:
    return RunIdentity(
        run_id=run_id,
        specification_fingerprint="spec",
        receipt_digest="receipt",
        sample_digest="sample",
        queue_namespace="test",
    )


def tasks():
    return [
        {"treatment_order": 1, "outcome_family": "population", "status": "pending"},
        {"treatment_order": 1, "outcome_family": "poi", "status": "pending"},
        {"treatment_order": 2, "outcome_family": "population", "status": "pending"},
        {"treatment_order": 3, "outcome_family": "population", "status": "gsc_pending"},
        {"treatment_order": 4, "outcome_family": "population", "status": "mc_pending"},
        {"treatment_order": 5, "outcome_family": "population", "status": "mc_labelled"},
    ]


def test_dynamic_claims_are_unique_and_order_locked(tmp_path):
    pool = RoundPool(tmp_path / "pool.sqlite", lease_seconds=60)
    pool.create_run(identity(), tasks())
    claimed = []
    barrier = threading.Barrier(3)

    def worker(name: str):
        barrier.wait()
        lease = pool.claim_next("run-test", "same_city_matching", name)
        if lease is not None:
            claimed.append((name, lease.treatment_order, lease.outcome_family))

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(claimed) == 2
    assert len(set(claimed)) == 2
    assert {item[1] for item in claimed} == {1, 2}
    # The second task for order 1 is blocked until the first lease is released.
    lease = pool.claim_next("run-test", "same_city_matching", "w3")
    assert lease is None


def test_round_barrier_and_forward_transition(tmp_path):
    pool = RoundPool(tmp_path / "pool.sqlite")
    pool.create_run(identity(), tasks())
    lease = pool.claim_next("run-test", "same_city_matching", "w0")
    assert lease is not None
    assert pool.claim_next("run-test", "same_city_gsc", "w1") is None
    assert pool.finish("run-test", lease, "w0", queue_status="gsc_pending")
    assert pool.close_round_if_drained("run-test", "same_city_matching") is False
    second = pool.claim_next("run-test", "same_city_matching", "w1")
    assert second is not None
    assert pool.finish("run-test", second, "w1", queue_status="matched_labelled")
    third = pool.claim_next("run-test", "same_city_matching", "w2")
    assert third is not None
    assert pool.finish("run-test", third, "w2", queue_status="matched_labelled")
    assert pool.close_round_if_drained("run-test", "same_city_matching")
    assert pool.round_states("run-test")[1]["state"] == "open"
    gsc_lease = pool.claim_next("run-test", "same_city_gsc", "w3")
    assert gsc_lease is not None
    assert gsc_lease.treatment_order == 1


def test_hard_failed_task_can_be_requeued(tmp_path):
    pool = RoundPool(tmp_path / "pool.sqlite")
    pool.create_run(
        identity(),
        [{"treatment_order": 1, "outcome_family": "population", "status": "pending"}],
    )
    lease = pool.claim_next("run-test", "same_city_matching", "w0")
    assert lease is not None
    assert pool.fail("run-test", lease, "w0", "boom", retry=False)
    assert pool.counts("run-test") == {"failed": 1}

    assert pool.requeue_failed("run-test") == 1
    again = pool.claim_next("run-test", "same_city_matching", "w1")
    assert again is not None
    assert again.treatment_order == 1
    assert pool.finish("run-test", again, "w1", queue_status="matched_labelled")
    assert pool.counts("run-test") == {"completed": 1}


def test_stale_lease_cannot_commit_after_expiry_and_reclaim(tmp_path):
    pool = RoundPool(tmp_path / "pool.sqlite", lease_seconds=0.01)
    pool.create_run(identity(), [{"treatment_order": 1, "outcome_family": "population", "status": "pending"}])
    old = pool.claim_next("run-test", "same_city_matching", "old")
    assert old is not None
    time.sleep(0.03)
    assert pool.requeue_expired("run-test") == 1
    new = pool.claim_next("run-test", "same_city_matching", "new")
    assert new is not None
    assert not pool.finish("run-test", old, "old", queue_status="matched_labelled")
    assert pool.finish("run-test", new, "new", queue_status="matched_labelled")
    assert pool.counts("run-test") == {"completed": 1}
