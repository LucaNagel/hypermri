// Control group "parameters": the measurement parameters of the scan
// (`_parameters`, rows of hypermri.visualization.scan_parameters, sent once).
// "show parameters" opens a table grouped as the scan's PARAMETER_SCHEMA;
// typing in the filter keeps the rows whose name, group or value contains
// the text. Open state and filter are kept in the browser only, and the
// group is not rebuilt with a new block, so the filter survives.
// "original names" (shown while the table is open) adds the vendor's own
// name of each parameter (row.source, for example PVM_Matrix) after its name.

function parameterControls(v) {
  const rows = v.model.get("_parameters") || [];
  if (!rows.length) return [];
  const show = element("input");
  show.type = "checkbox";
  show.checked = Boolean(v.showParameters);
  show.title = "Show the measurement parameters of the scan (scan.get_parameters())";
  const filter = element("input");
  filter.type = "search";
  filter.placeholder = "filter";
  filter.value = v.parameterFilter || "";
  filter.size = 14;
  filter.hidden = !show.checked;
  filter.title = "Keep rows whose group, name or value contains this text";
  const original = element("input");
  original.type = "checkbox";
  original.checked = Boolean(v.showParameterSources);
  original.title = "Show the scanner's own name of each parameter, for example PVM_Matrix";
  const originalLabel = labelled("original names", original);
  originalLabel.hidden = !show.checked;
  const scroller = element("div", "hm-csi-parameter-scroll");
  scroller.hidden = !show.checked;

  const fill = () => {
    const needle = filter.value.trim().toLowerCase();
    const table = element("table", "hm-csi-parameter-table");
    let group = null;
    for (const row of rows) {
      const source = original.checked ? row.source || "" : "";
      const text = `${row.group} ${row.name} ${source} ${row.value}`.toLowerCase();
      if (needle && !text.includes(needle)) continue;
      if (row.group !== group) {
        group = row.group;
        const heading = element("tr", "hm-csi-parameter-group");
        const cell = element("th", "", group);
        cell.colSpan = 2;
        heading.append(cell);
        table.append(heading);
      }
      const line = element("tr");
      // Name and source each stay on one line; in a narrow view the source
      // moves below the name, so the value column stays in sight.
      const name = element("td", "hm-csi-parameter-name");
      name.append(element("span", "", row.name));
      if (source) name.append(" ", element("span", "hm-csi-parameter-source", source));
      line.append(name, element("td", "", row.value));
      table.append(line);
    }
    scroller.replaceChildren(table);
  };
  show.addEventListener("change", () => {
    v.showParameters = show.checked;
    filter.hidden = !show.checked;
    originalLabel.hidden = !show.checked;
    scroller.hidden = !show.checked;
  });
  original.addEventListener("change", () => {
    v.showParameterSources = original.checked;
    fill();
  });
  filter.addEventListener("input", () => {
    v.parameterFilter = filter.value;
    fill();
  });
  fill();
  // A label holds one control, so the three sit side by side in a span.
  const head = element("span", "hm-csi-parameter-head");
  head.append(labelled("show parameters", show), originalLabel, filter);
  return [head, scroller];
}
