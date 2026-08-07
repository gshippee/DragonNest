(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.DragonNestAdminSelection = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function sortedRecentTasks(tasks, limit) {
    const maximum = Math.max(1, Number(limit) || 50);
    return [...(tasks || [])]
      .sort((left, right) => Number(right.created_at || 0) - Number(left.created_at || 0))
      .slice(0, maximum);
  }

  function reconcileSelection(tasks, selectedTaskId, followLatest) {
    const recent = sortedRecentTasks(tasks, 50);
    if (!recent.length) return { tasks: recent, selectedTaskId: "" };
    if (followLatest) return { tasks: recent, selectedTaskId: recent[0].task_id };
    const pinned = recent.find((task) => task.task_id === selectedTaskId);
    return { tasks: recent, selectedTaskId: pinned ? pinned.task_id : recent[0].task_id };
  }

  return { sortedRecentTasks, reconcileSelection };
});
