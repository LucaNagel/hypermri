// Control group "time_course": what the time panel shows. Empty when there
// is no time course.
//
// The spectral range is chosen by dragging in the spectrum or with the two
// inputs (in the displayed spectral unit); "follow point" shows the value at
// the selected spectral point again. "mask mean" adds the mean inside the
// chosen mask. All changes go to Python as a `{type: "time_course"}` message.

function sendTimeCourse(v, content) {
  v.model.send({ type: "time_course", ...content });
}

function timeCourseControls(v) {
  const time = v.view.time;
  const course = v.timeCourse;
  if (!time || !course.range) return [];
  const items = [element("span", "hm-csi-control-label", "Time course")];
  if (time.choices.length > 1) {
    const dim = select(time.choices, time.dim);
    dim.addEventListener("change", () => sendTimeCourse(v, { dim: dim.value }));
    items.push(labelled("along", dim));
  }
  const [first, last] = course.range_values;
  // The other limit is read when one changes; follow mode starts from the point.
  const lower = limitInput(v, first, (value) => {
    sendTimeCourse(v, { range: [value, upper.valueAsNumber] });
    return true;
  }, "Enter a number in the spectral unit.");
  const upper = limitInput(v, last, (value) => {
    sendTimeCourse(v, { range: [lower.valueAsNumber, value] });
    return true;
  }, "Enter a number in the spectral unit.");
  const unit = v.view.curve_label ? ` [${v.view.curve_label}]` : "";
  const follow = button("Follow point", "Show the value at the selected spectral point (double-click in the spectrum)");
  follow.disabled = course.follows_point;
  follow.addEventListener("click", () => sendTimeCourse(v, { range: null }));
  const maskBox = element("input");
  maskBox.type = "checkbox";
  maskBox.checked = Boolean(course.mask);
  maskBox.disabled = !v.activeMask;
  maskBox.title = "Add the mean time course inside the chosen mask (shown slice only)";
  maskBox.addEventListener("change", () =>
    sendTimeCourse(v, { mask: maskBox.checked ? v.activeMask : null }),
  );
  items.push(
    labelled(`from${unit}`, lower),
    labelled("to", upper),
    follow,
    labelled("mask mean", maskBox),
  );
  if (v.timeMessage) items.push(element("span", "hm-csi-mask-error", v.timeMessage));
  return items;
}
