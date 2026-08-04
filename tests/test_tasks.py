from dragon_nest.tasks import (
    AttemptState,
    ResultDisposition,
    TaskState,
    TaskStore,
)


def test_cancellation_is_final_and_late_result_is_stale():
    tasks = TaskStore()
    task = tasks.create("request", task_id="task-cancel")
    attempt = tasks.assign(task.task_id, "pc-01")
    tasks.mark_running(attempt.attempt_id)

    tasks.begin_cancellation(task.task_id)
    cancelled = tasks.complete_cancellation(task.task_id)
    disposition = tasks.record_result(attempt.attempt_id, "too late")
    final = tasks.get(task.task_id)

    assert cancelled.state == TaskState.CANCELLED
    assert cancelled.attempts[0].state == AttemptState.CANCELLED
    assert disposition == ResultDisposition.STALE
    assert final.state == TaskState.CANCELLED
    assert final.result is None
    assert final.stale_results[-1].result == "too late"


def test_failed_execution_sets_controlled_task_failure():
    tasks = TaskStore()
    task = tasks.create("request")
    attempt = tasks.assign(task.task_id, "pc-01")
    tasks.mark_running(attempt.attempt_id)

    disposition = tasks.record_result(
        attempt.attempt_id,
        None,
        success=False,
        error_code="RUNTIME_ERROR",
        error_message="executor failed",
    )
    failed = tasks.get(task.task_id)

    assert disposition == ResultDisposition.FAILED
    assert failed.state == TaskState.FAILED
    assert failed.error_code == "RUNTIME_ERROR"
    assert failed.attempts[0].state == AttemptState.FAILED


def test_replica_winner_cancels_loser_and_late_result_is_stale():
    tasks = TaskStore()
    task = tasks.create("request", task_id="task-race")
    fast = tasks.assign_replica(task.task_id, "pc-01")
    slow = tasks.assign_replica(task.task_id, "phone-01")
    tasks.mark_replica_running(fast.attempt_id)
    tasks.mark_replica_running(slow.attempt_id)

    disposition = tasks.record_replica_result(fast.attempt_id, "winner")
    late_disposition = tasks.record_result(slow.attempt_id, "too late")
    final = tasks.get(task.task_id)

    assert disposition == ResultDisposition.ACCEPTED
    assert late_disposition == ResultDisposition.STALE
    assert final.state == TaskState.SUCCEEDED
    assert final.accepted_attempt_id == fast.attempt_id
    assert final.attempts[0].state == AttemptState.SUCCEEDED
    assert final.attempts[1].state == AttemptState.CANCELLED
    assert final.stale_results[-1].attempt_id == slow.attempt_id
    assert final.result == "winner"
