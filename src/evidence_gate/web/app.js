"use strict";

// Report strings are always text nodes, including model output and documents.
// No remote assets, eval(), HTML injection, or experiment mutation is used.
const $ = (selector) => document.querySelector(selector);
const state = { report: null, method: 0, selected: null, query: "", scenario: "", result: "" };
const actionNames = { answer: "作答", query: "继续查询", search: "继续查询", abstain: "拒答", ask: "请求补充", clarify: "请求澄清", escalate: "转人工" };
const metricNames = { action_correct: "动作正确", answer_correct: "可答目标通过", citation_correct: "引用集合正确", format_valid: "格式有效", success: "整体通过" };
const scenarioNames = {
  current: "有效证据", valid: "有效证据", supported: "有效证据", stale: "过期证据", expired: "过期证据",
  missing: "缺少证据", no_evidence: "缺少证据", conflict: "证据冲突", conflicting: "证据冲突",
  irrelevant: "无关证据", budget_zero: "预算耗尽", exhausted: "预算耗尽", outdated: "过期证据",
  value_changed: "证据值改变", distractor: "干扰证据", reordered: "证据顺序改变", missing_search: "缺少依据 · 可查询",
  missing_stop: "缺少依据 · 无预算", expired_search: "已过期 · 可查询", expired_stop: "已过期 · 无预算", future_search: "未生效 · 可查询",
  conflict_search: "证据冲突 · 可查询", conflict_stop: "证据冲突 · 无预算", ambiguous: "问题范围不明确",
  untrusted_search: "非可信来源 · 可查询", untrusted_stop: "非可信来源 · 无预算",
  future: "未生效证据", untrusted: "非可信来源", irrelevant_topic: "无关主题干扰", consensus: "多份官方证据一致",
};
const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value, fallback = "未记录") => value === undefined || value === null || value === "" ? fallback : typeof value === "object" ? JSON.stringify(value) : String(value);
const pretty = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "未记录";
const list = (value) => Array.isArray(value) ? value : [];
function scenarioName(value) {
  if (scenarioNames[value]) return scenarioNames[value];
  const paired = typeof value === "string" && value.match(/^(.+)_b([01])$/);
  return paired && scenarioNames[paired[1]]
    ? `${scenarioNames[paired[1]]} · 查询预算 ${paired[2]}` : text(value, "未标注场景");
}
const actionName = (value) => actionNames[value] || "报告原始动作";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "onClick") node.addEventListener("click", value);
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
function badge(label, kind = "") { return el("span", { class: `badge ${kind}` }, label); }
function statusBadge(metrics) {
  return metrics?.success === true ? badge("通过", "pass") : metrics?.success === false ? badge("未通过", "fail") : badge("未记录", "unknown");
}
function empty(title, description) { return el("div", { class: "empty-state" }, el("strong", {}, title), description); }
function method() { return state.report.methods[state.method] || {}; }
function cases() { return list(method().cases).filter(isRecord); }
function knownKind(value) { return value === "model" ? "模型实测" : value === "rule" ? "规则基线" : "类型未记录"; }
function firstValue(object, keys) { for (const key of keys) if (object?.[key] !== undefined && object[key] !== null) return object[key]; return undefined; }
function selectMethod(index, caseId = null) {
  state.method = index;
  state.selected = cases().find(item => item.id === caseId) || cases()[0] || null;
  state.scenario = "";
  state.query = "";
  state.result = "";
  $("#method-select").value = String(index);
  $("#case-search").value = "";
  $("#result-select").value = "";
  const variants = [...new Set(cases().map(item => text(item.variant, "")))].sort();
  $("#scenario-select").replaceChildren(el("option", { value: "" }, "全部场景"), ...variants.filter(Boolean).map(value => el("option", { value }, scenarioName(value))));
  $("#method-kind").className = `badge ${method().kind === "model" ? "model" : method().kind === "rule" ? "rule" : "unknown"}`;
  $("#method-kind").textContent = knownKind(method().kind);
  $("#method-note").textContent = method().kind === "rule" ? "规则输出单独呈现，不计作模型训练效果。" : method().kind === "model" ? "模型输出与评测结论均读取自已保存的实验记录。" : "此方法未声明类型，请在来源中核验执行方式。";
  renderMetrics();
  renderCases();
}
function buildMetricCards(summary = {}) {
  // Denominators and rates are taken from the saved evaluator summary. In
  // particular, an unparseable output is not a risk-free emitted answer.
  const percent = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? `${(100 * value).toFixed(1)}%` : "—";
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const n = count(summary.n), groups = count(summary.groups);
  const emitted = count(summary.denominators?.emitted_answers);
  const answerable = count(summary.denominators?.answerable);
  const unanswerable = count(summary.denominators?.unanswerable);
  const format = percent(summary.format_valid_rate);
  return [
    { label: "整体成功率", value: percent(summary.success_rate), note: n === null ? "案例分母未记录" : `${n} 个评测案例；动作、答案、引用均正确` },
    { label: "可答成功率", value: answerable === 0 ? "不适用" : percent(summary.answerable_success), note: answerable === null ? "可答案例分母未记录" : `${answerable} 个本应作答的案例为分母` },
    { label: "整组全通过率", value: percent(summary.all_variants_correct_rate), note: groups === null ? "问题组分母未记录" : `${groups} 个问题组；组内全部变体均须通过` },
    { label: "动作正确率", value: percent(summary.action_accuracy), note: n === null ? "案例分母未记录" : `${n} 个案例；只检查决策动作` },
    { label: "动作 Macro-F1", value: percent(summary.action_macro_f1), note: "四种动作的 F1 等权平均；不能替代可答成功率" },
    { label: "不应作答却作答比例", value: unanswerable === 0 ? "不适用" : unanswerable === null ? "—" : percent(summary.over_answer_rate), note: unanswerable === null ? "不应作答案例分母未记录" : `${unanswerable} 个本不应作答的案例为分母；低值不代表模型愿意正确作答` },
    { label: "格式有效率", value: format, note: n === null ? "案例分母未记录" : `${n} 个案例；不可解析的输出计为格式失败` },
    { label: "证据合同未通过的回答比例", value: emitted === 0 ? "不可估计" : emitted === null ? "—" : percent(summary.unsupported_answer_rate),
      note: emitted === 0 ? "没有可解析的 answer；格式失败不能视为安全" : emitted === null ? "未记录可解析 answer 的分母，无法估计风险" : `${emitted} 次可解析 answer 为分母；含不应回答时回答、错值、错引或缺引；格式有效率 ${format}` },
  ];
}
function renderMetrics() {
  const cards = buildMetricCards(isRecord(method().summary) ? method().summary : {});
  $("#metrics").replaceChildren(...cards.map(card => el("div", { class: "metric-card" }, el("span", { class: "metric-label" }, card.label), el("div", { class: "metric-value" }, card.value, card.unit ? el("small", {}, card.unit) : null), el("span", { class: "metric-note" }, card.note))));
}
function filteredCases() {
  const query = state.query.trim().toLocaleLowerCase();
  return cases().filter(item => {
    const haystack = `${text(item.id, "")} ${text(item.observation?.question, "")} ${text(item.group_id, "")}`.toLocaleLowerCase();
    return (!query || haystack.includes(query)) && (!state.scenario || item.variant === state.scenario)
      && (!state.result || (state.result === "passed" && item.metrics?.success === true) || (state.result === "failed" && item.metrics?.success === false) || (state.result === "unknown" && typeof item.metrics?.success !== "boolean"));
  });
}
function renderCases() {
  const visible = filteredCases();
  if (!visible.includes(state.selected)) state.selected = visible[0] || null;
  $("#filtered-count").textContent = `${visible.length} / ${cases().length}`;
  $("#case-list").replaceChildren(...(visible.length ? visible.map(item => el("button", {
    type: "button", class: `case-row${item === state.selected ? " selected" : ""}`, "aria-pressed": item === state.selected,
    onClick: event => selectCase(item, event.currentTarget),
  }, el("span", { class: "case-row-top" }, el("span", { class: "case-id", title: text(item.id) }, text(item.id)), statusBadge(item.metrics)),
  el("span", { class: "case-question" }, text(item.observation?.question, "该案例未记录问题")),
  el("span", { class: "case-row-bottom" }, el("span", {}, scenarioName(item.variant)), el("span", { class: "case-action" }, `${text(item.expected?.action, "?")} → ${text(item.prediction?.action, "?")}`)))) : [empty("没有匹配的案例", "调整场景、结果筛选或搜索词后重试。") ]));
  renderDetail();
}
function selectCase(item, row) {
  // Keep the actual button (and its keyboard focus/scroll position) alive.
  // Selecting one case must not rebuild the entire long filtered list.
  const previous = $("#case-list .case-row.selected");
  if (previous) { previous.classList.remove("selected"); previous.setAttribute("aria-pressed", "false"); }
  row.classList.add("selected");
  row.setAttribute("aria-pressed", "true");
  state.selected = item;
  renderDetail();
}
function section(index, title, content, trailing = null) {
  return el("section", { class: "detail-section" }, el("div", { class: "section-title" }, el("h3", {}, el("span", { class: "section-index" }, index), title), trailing), content);
}
function evidenceCard(document) {
  return el("div", { class: "evidence-card" }, el("div", { class: "evidence-card-top" }, el("span", { class: "evidence-card-id" }, text(document.id)), badge("输入证据")),
    el("div", { class: "evidence-source" }, `来源：${text(document.source)}`), el("p", {}, text(document.text, "未记录正文")),
    el("div", { class: "evidence-date" }, `有效期 ${text(document.valid_from, "未记录起点")} → ${text(document.valid_until, "未记录终点")}`));
}
function decisionCard(value, title, prediction, kind = method().kind) {
  const citations = list(value?.citations);
  return el("div", { class: `decision-card${prediction ? " prediction" : ""}` },
    el("div", { class: "decision-card-header" }, title, badge(prediction ? knownKind(kind) : "监督目标", prediction ? kind : "")),
    el("div", { class: "decision-body" }, el("div", { class: "decision-action" }, el("span", { class: "action-tag" }, text(value?.action, "未解析")), el("span", { class: "action-explanation" }, value?.action ? actionName(value.action) : "报告中没有有效预测对象")),
      el("div", { class: "decision-value" }, text(value?.value, value?.action === "answer" ? "未记录答案" : "该动作无答案值")),
      el("div", { class: "citation-line" }, `引用（${citations.length}）`, ...(citations.length ? citations.map(item => el("span", { class: "citation-chip" }, text(item))) : [el("span", {}, "无")]))));
}
function answerRiskStatus(item) {
  const metrics = isRecord(item.metrics) ? item.metrics : {};
  if (metrics.format_valid === false) return { kind: "", label: "· 作答依据检查：格式无效，无法估计" };
  if (item.prediction && item.prediction.action !== "answer") return { kind: "", label: "· 作答依据检查：本步没有作答" };
  if (item.prediction?.action !== "answer" || metrics.format_valid !== true) return { kind: "", label: "· 作答依据检查：未记录" };
  if (metrics.unsupported_answer === true) return { kind: "fail", label: "× 作答依据检查失败（含答案值或引用集合错误）" };
  if (metrics.unsupported_answer === false) return { kind: "pass", label: "✓ 作答依据检查通过（限本评测规则）" };
  return { kind: "", label: "· 作答依据检查：未记录" };
}
function renderDetail() {
  const item = state.selected;
  if (!item) { $("#case-detail").replaceChildren(empty("选择一个案例开始回放", "这里将并排呈现输入证据、监督目标、实际输出与评测标记。")); return; }
  const observation = isRecord(item.observation) ? item.observation : {};
  const documents = list(observation.documents).filter(isRecord);
  const metrics = isRecord(item.metrics) ? item.metrics : {};
  const checks = Object.entries(metricNames).map(([key, label]) => {
    const kind = metrics[key] === true ? "pass" : metrics[key] === false ? "fail" : "";
    return el("span", { class: `check-item ${kind}` }, el("span", { class: "check-symbol", "aria-hidden": true }, kind === "pass" ? "✓" : kind === "fail" ? "×" : "·"), `${label}${kind ? "" : metrics[key] === null ? "：不适用" : "：未记录"}`);
  });
  const risk = answerRiskStatus(item);
  checks.push(el("span", { class: `check-item ${risk.kind}` }, risk.label));
  const failures = [];
  for (const [key, label] of Object.entries(metricNames)) if (key !== "success" && metrics[key] === false) failures.push(label.replace("正确", "不正确").replace("有效", "无效").replace("通过", "未通过"));
  if (metrics.unsupported_answer === true) failures.push("作答依据检查失败；不应作答、答案值错误或引用集合不完整/不匹配均计入");
  const explicitReason = firstValue(item, ["failure_reason", "failure_reasons", "error", "errors"]);
  const note = explicitReason ? `报告记录：${text(explicitReason)}` : failures.length ? `评测标记：${failures.join("；")}。结合原始输出核查原因。` : metrics.success === true ? "该案例通过了报告定义的成功检查；结论仅适用于当前输入与评测规则。" : "报告未给出明确失败原因，请结合原始输出和评测实现检查。";
  const outputTitle = method().kind === "rule" ? "规则实际输出" : method().kind === "model" ? "模型实际输出" : "方法实际输出";
  const decision = el("div", {}, el("div", { class: "decision-grid" }, decisionCard(item.expected, "期望动作与答案", false), decisionCard(item.prediction, outputTitle, true)),
    el("div", { class: "checklist" }, checks), el("p", { class: `failure-note${metrics.success === true ? " success" : ""}` }, note),
    el("details", { class: "raw-output" }, el("summary", {}, `${method().kind === "rule" ? "规则" : method().kind === "model" ? "模型" : "方法"}原始输出 · 未加工文本`), el("pre", {}, pretty(item.raw))));
  const batchSize = item.generation?.batch_size;
  const latencyLabel = method().kind === "rule" ? "规则执行耗时"
    : method().kind === "model" && Number.isInteger(batchSize) && batchSize > 1 ? "批量均摊耗时（非单请求）"
    : method().kind === "model" && batchSize === 1 ? "单例生成耗时（不含加载）"
    : "记录耗时（口径未记录）";
  const header = el("div", { class: "detail-head" }, el("div", { class: "detail-kicker" }, badge(scenarioName(item.variant)), statusBadge(metrics), el("code", {}, text(item.id))),
    el("h2", {}, text(observation.question, "问题未记录")), el("div", { class: "detail-meta" },
      el("span", {}, "判断日期", el("strong", {}, text(observation.as_of))),
      el("span", {}, "剩余查询预算", el("strong", {}, text(observation.budget))),
      el("span", {}, latencyLabel, el("strong", {}, typeof item.latency_ms === "number" ? `${item.latency_ms.toFixed(1)} ms` : "未记录")),
      el("span", {}, "问题组", el("strong", {}, text(item.group_id)))));
  $("#case-detail").replaceChildren(header,
    section("01", "模型可见的证据", documents.length ? el("div", { class: "evidence-grid" }, documents.map(evidenceCard)) : empty("当前输入没有证据文档", "是否继续查询或拒答，应结合问题、预算与监督目标判断。"), badge(`${documents.length} 份文档`)),
    section("02", "决策与评测", decision),
    section("03", "同问题反事实对比", counterfactuals(item)),
    section("04", "同输入的方法对照", methodComparison(item)));
}
function counterfactuals(item) {
  const siblings = item.group_id ? cases().filter(other => other.group_id === item.group_id) : [item];
  if (siblings.length < 2) return el("p", { class: "section-caption" }, "本报告没有记录该问题的其他输入变体；不推算反事实结果。");
  const table = el("table", { class: "comparison-table" }, el("thead", {}, el("tr", {}, ...["输入场景", "证据 / 预算", "期望动作", "实际动作", "结果"].map(label => el("th", { scope: "col" }, label)))),
    el("tbody", {}, siblings.map(other => el("tr", { class: other === item ? "current-row" : "" },
      el("td", {}, el("button", { type: "button", onClick: () => {
        state.selected = other; state.scenario = ""; state.result = ""; state.query = "";
        $("#scenario-select").value = ""; $("#result-select").value = ""; $("#case-search").value = ""; renderCases();
      } }, scenarioName(other.variant)), el("span", { class: "case-id" }, text(other.id))),
      el("td", {}, `${list(other.observation?.documents).length} 份 / ${text(other.observation?.budget, "?")}`), el("td", {}, text(other.expected?.action)), el("td", {}, text(other.prediction?.action, "未解析")), el("td", {}, statusBadge(other.metrics))))));
  return el("div", {}, el("div", { class: "table-scroll" }, table), el("p", { class: "section-caption" }, "相同问题组、相同方法，仅比较报告中实际存在的输入变体。点击场景可展开完整证据。"));
}
function methodComparison(item) {
  const canonical = value => Array.isArray(value) ? value.map(canonical) : isRecord(value) ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
  const observation = JSON.stringify(canonical(item.observation));
  const matches = state.report.methods.map((candidate, index) => ({ method: candidate, index, case: list(candidate.cases).find(other => other?.id === item.id && JSON.stringify(canonical(other.observation)) === observation) })).filter(match => match.case);
  if (matches.length < 2) return el("p", { class: "section-caption" }, "当前案例只记录了一种方法，无法进行跨方法对照。");
  return el("div", { class: "table-scroll" }, el("table", { class: "comparison-table" }, el("thead", {}, el("tr", {}, ...["方法", "执行类型", "实际动作", "结果"].map(label => el("th", { scope: "col" }, label)))),
    el("tbody", {}, matches.map(match => el("tr", { class: match.index === state.method ? "current-row" : "" },
      el("td", {}, el("button", { type: "button", onClick: () => selectMethod(match.index, item.id) }, text(match.method.name))), el("td", {}, badge(knownKind(match.method.kind), match.method.kind)), el("td", {}, text(match.case.prediction?.action, "未解析")), el("td", {}, statusBadge(match.case.metrics)))))));
}

function svg(tag, attrs = {}, content = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (content !== null) node.textContent = content;
  return node;
}
function trainingRuns(report = state.report) {
  const training = list(report.training).map((entry, index) => Array.isArray(entry) ? { name: `训练记录 ${index + 1}`, history: entry } : entry).filter(isRecord);
  const runs = [];
  const flat = new Map();
  for (const entry of training) {
    const nested = firstValue(entry, ["history", "logs", "records", "losses", "steps"]);
    if (Array.isArray(nested)) {
      runs.push({ name: text(entry.name ?? entry.method ?? entry.model, "训练记录"), records: nested.filter(isRecord), metadata: Object.fromEntries(Object.entries(entry).filter(([key]) => !["history", "logs", "records", "losses", "steps"].includes(key))) });
    } else {
      const name = text(entry.method ?? entry.name, "训练记录");
      if (!flat.has(name)) flat.set(name, []);
      flat.get(name).push(entry);
    }
  }
  for (const [name, records] of flat) runs.push({ name, records, metadata: {} });
  return runs;
}
function lossPoints(records) {
  return records.map(record => ({ step: firstValue(record, ["step", "iteration", "global_step", "iter"]), loss: firstValue(record, ["loss", "train_loss", "training_loss"]), record })).filter(point => typeof point.step === "number" && Number.isFinite(point.step) && typeof point.loss === "number" && Number.isFinite(point.loss)).sort((a, b) => a.step - b.step);
}
function trainingEvidence(run) {
  const metadata = isRecord(run.metadata?.metadata) ? run.metadata.metadata : run.metadata || {};
  const step = value => Number.isInteger(value) && value >= 0 ? value : null;
  const selected = step(metadata.selected_step);
  const completed = step(metadata.completed_training_steps ?? (selected === null ? metadata.optimizer_steps : undefined));
  const budget = step(metadata.total_training_budget_steps);
  const points = lossPoints(run.records);
  const fields = [
    { label: "所选检查点", value: selected === null ? "未记录" : `step ${selected}` },
    { label: "本轮已完成训练", value: completed === null ? "未记录" : `${completed} steps` },
    { label: "本轮训练总预算", value: budget === null ? "未记录" : `${budget} steps` },
  ];
  let note = selected === null ? "曲线仅呈现报告保存的训练记录；不从名称或记录条数推断选模步骤与总预算。"
    : `曲线对应所选 step ${selected} 的训练前缀；总预算与已完成步数另列，后续训练损失不归属于该检查点。`;
  if (selected !== null && points.length && points[points.length - 1].step !== selected) note = "记录不一致：曲线末步与所选检查点不同，请核查原始训练元数据；不能据此归属训练效果。";
  return { fields, note };
}
function formatLoss(value, decimalPlaces = 4) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "未记录";
  if (value === 0) return "0";
  return Math.abs(value) < 10 ** -decimalPlaces || Math.abs(value) >= 10000
    ? value.toExponential(3) : value.toFixed(decimalPlaces);
}
function lossChart(points, name) {
  const width = 950, height = 300, left = 56, right = 22, top = 20, bottom = 42;
  const minStep = Math.min(...points.map(point => point.step)), maxStep = Math.max(...points.map(point => point.step));
  const minLoss = Math.min(...points.map(point => point.loss)), maxLoss = Math.max(...points.map(point => point.loss));
  const span = maxLoss - minLoss || Math.max(Math.abs(maxLoss) * 0.1, 0.1);
  const low = Math.max(0, minLoss - span * 0.15), high = maxLoss + span * 0.15;
  const x = step => left + (maxStep === minStep ? 0.5 : (step - minStep) / (maxStep - minStep)) * (width - left - right);
  const y = loss => top + (high - loss) / (high - low || 1) * (height - top - bottom);
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "loss-chart", role: "img", "aria-label": `${name} 训练损失曲线，共 ${points.length} 个记录点` });
  chart.append(svg("title", {}, `${name}：训练损失`));
  chart.append(svg("desc", {}, `横轴为已记录 step，纵轴为 loss。初始记录 ${points[0].loss}，最终记录 ${points[points.length - 1].loss}；损失不等于任务成功率。`));
  for (let tick = 0; tick <= 4; tick++) {
    const loss = low + (high - low) * tick / 4, yy = y(loss);
    chart.append(svg("line", { x1: left, x2: width - right, y1: yy, y2: yy, class: "chart-grid" }), svg("text", { x: left - 10, y: yy + 4, "text-anchor": "end", class: "chart-label" }, formatLoss(loss, 2)));
    const step = minStep + (maxStep - minStep) * tick / 4;
    if (maxStep !== minStep || tick === 2) chart.append(svg("text", { x: x(step), y: height - 17, "text-anchor": "middle", class: "chart-label" }, Number.isInteger(step) ? String(step) : step.toFixed(1)));
  }
  const coordinates = points.map(point => `${x(point.step)},${y(point.loss)}`).join(" ");
  if (points.length > 1) chart.append(svg("polygon", { points: `${x(points[0].step)},${height - bottom} ${coordinates} ${x(points[points.length - 1].step)},${height - bottom}`, class: "chart-area" }));
  chart.append(svg("polyline", { points: coordinates, class: "chart-line" }));
  for (const point of points) { const circle = svg("circle", { cx: x(point.step), cy: y(point.loss), r: points.length > 100 ? 2 : 3.5, class: "chart-point" }); circle.append(svg("title", {}, `step ${point.step} · loss ${point.loss}`)); chart.append(circle); }
  chart.append(svg("text", { x: left, y: 10, class: "chart-label" }, "Loss"), svg("text", { x: width - right, y: height - 1, "text-anchor": "end", class: "chart-label" }, "Step"));
  return chart;
}
function renderTraining() {
  const runs = trainingRuns();
  if (!runs.length) { $("#training-content").replaceChildren(el("div", { class: "training-card" }, empty("这份报告没有训练记录", "评测输出不等于已完成微调。请先查看来源与边界，并在实际训练后保存 step / loss 记录。"))); return; }
  $("#training-content").replaceChildren(...runs.map(run => {
    const points = lossPoints(run.records);
    const evidence = trainingEvidence(run);
    return el("article", { class: "training-card" }, el("div", { class: "panel-heading" }, el("h3", {}, run.name), badge(`${run.records.length} 条记录`)),
      el("div", { class: "training-scope" }, el("div", { class: "training-scope-fields" }, evidence.fields.map(field => el("span", {}, field.label, el("strong", {}, field.value)))), el("p", { class: "training-scope-note" }, evidence.note)),
      points.length ? el("div", { class: "chart-wrap" }, lossChart(points, run.name)) : empty("没有可绘制的 step / loss", "原始训练记录保留如下，未将缺失值当作零。"),
      points.length ? el("div", { class: "chart-legend" }, el("span", {}, "首个 loss", el("strong", {}, formatLoss(points[0].loss))), el("span", {}, "末个 loss", el("strong", {}, formatLoss(points[points.length - 1].loss))), el("span", {}, "记录步范围", el("strong", {}, `${points[0].step} – ${points[points.length - 1].step}`)), el("span", {}, "训练损失变化不代表任务能力提升")) : null,
      el("details", { class: "raw-output training-records" }, el("summary", {}, "查看完整训练记录与元数据"), el("pre", {}, pretty({ metadata: run.metadata, records: run.records }))));
  }));
}
function safeLink(value) {
  try { const parsed = new URL(value); return ["https:", "http:"].includes(parsed.protocol) ? parsed.href : null; } catch { return null; }
}
function sourceEntries(provenance) {
  const source = firstValue(provenance, ["sources", "references", "repositories"]) ?? state.report.sources;
  return Array.isArray(source) ? source : isRecord(source) ? Object.entries(source).map(([name, value]) => isRecord(value) ? { name, ...value } : { name, url: value }) : source ? [source] : [];
}
function renderProvenance() {
  const provenance = isRecord(state.report.provenance) ? state.report.provenance : {};
  const sources = sourceEntries(provenance);
  const limits = firstValue(provenance, ["limitations", "limits", "caveats"]) ?? state.report.limitations;
  const limitList = Array.isArray(limits) ? limits : limits ? [limits] : [];
  const table = el("table", { class: "provenance-table" }, el("tbody", {}, Object.entries(provenance).map(([key, value]) => el("tr", {}, el("td", {}, key), el("td", {}, typeof value === "object" ? el("pre", {}, pretty(value)) : text(value))))));
  const sourcesNode = sources.length ? el("ul", { class: "source-list" }, sources.map(source => {
    const record = isRecord(source) ? source : { url: source };
    const href = safeLink(record.url ?? record.link ?? record.repository);
    const name = text(record.name ?? record.title ?? record.url ?? record.link, "来源记录");
    return el("li", {}, href ? el("a", { href, target: "_blank", rel: "noopener noreferrer" }, `${name} ↗`) : el("span", {}, name), record.note || record.description || record.license ? el("p", {}, text(record.note ?? record.description ?? record.license)) : null);
  })) : el("p", { class: "provenance-note" }, "此报告未列出来源链接。请结合仓库文档、数据生成配置与模型卡进一步核验。");
  const left = el("section", { class: "provenance-card" }, el("div", { class: "panel-heading" }, el("h3", {}, "实验来源记录"), badge("报告原始字段")), el("div", { class: "provenance-card-content" }, Object.keys(provenance).length ? table : empty("未记录 provenance", "不能从界面推断模型、硬件、数据来源或训练是否完成。")));
  const right = el("div", {}, el("section", { class: "provenance-card" }, el("div", { class: "panel-heading" }, el("h3", {}, "参考与来源入口")), el("div", { class: "provenance-card-content" }, sourcesNode)),
    el("section", { class: "provenance-card" }, el("div", { class: "panel-heading" }, el("h3", {}, "实验边界")), el("div", { class: "provenance-card-content" }, limitList.length ? el("ul", { class: "boundary-list" }, limitList.map(limit => el("li", {}, text(limit)))) : el("p", { class: "provenance-note" }, "报告没有记录额外限制。未记录不代表没有限制；请查看模型、数据分布和评测定义。"))),
    el("section", { class: "provenance-card" }, el("div", { class: "panel-heading" }, el("h3", {}, "如何阅读这份回放")), el("div", { class: "provenance-card-content" }, el("ul", { class: "boundary-list" }, el("li", {}, "规则基线与模型实测有独立标识，不能把规则成功率当成微调效果。"), el("li", {}, "原始输出保留模型实际返回的文本；界面不修复 JSON，不重新生成答案。"), el("li", {}, "反事实对比只使用同一问题组的已记录输入，指标不外推到真实业务。"), el("li", {}, "本地服务在启动时读取报告快照，刷新页面不会重新运行训练或评测。")))));
  $("#provenance-content").replaceChildren(el("div", { class: "provenance-layout" }, left, right));
}
function switchView(name) {
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  }
  for (const view of ["cases", "training", "provenance"]) $(`#view-${view}`).hidden = view !== name;
}
async function init() {
  try {
    const response = await fetch("/api/report", { credentials: "same-origin", cache: "no-store" });
    if (!response.ok) throw new Error(`报告请求失败（HTTP ${response.status}）`);
    const report = await response.json();
    if (!isRecord(report) || !Array.isArray(report.methods) || report.methods.some(candidate => !isRecord(candidate))) throw new Error("报告结构无效：需要 methods 数组。");
    state.report = report;
    const provenance = isRecord(report.provenance) ? report.provenance : {};
    $("#generated-at").textContent = text(report.generated_at);
    if (report.generated_at) $("#generated-at").dateTime = String(report.generated_at);
    $("#model-label").textContent = text(firstValue(provenance, ["model", "model_id", "base_model"]));
    $("#device-label").textContent = text(firstValue(provenance, ["device", "hardware", "runtime"]));
    $("#data-label").textContent = text(firstValue(provenance, ["dataset", "data", "dataset_name"]));
    const models = report.methods.filter(candidate => candidate.kind === "model").length;
    const rules = report.methods.filter(candidate => candidate.kind === "rule").length;
    $("#method-count").textContent = `${models} 个模型 · ${rules} 个规则${report.methods.length - models - rules ? ` · ${report.methods.length - models - rules} 个未标注` : ""}`;
    $("#total-count").textContent = String(new Set(report.methods.flatMap(candidate => list(candidate.cases).filter(isRecord).map(item => item.id)).filter(Boolean)).size);
    $("#method-select").replaceChildren(...report.methods.map((candidate, index) => el("option", { value: index }, text(candidate.name, `方法 ${index + 1}`))));
    $("#method-select").disabled = report.methods.length === 0;
    $("#method-select").addEventListener("change", event => selectMethod(Number(event.target.value)));
    $("#case-search").addEventListener("input", event => { state.query = event.target.value; renderCases(); });
    $("#scenario-select").addEventListener("change", event => { state.scenario = event.target.value; renderCases(); });
    $("#result-select").addEventListener("change", event => { state.result = event.target.value; renderCases(); });
    for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => switchView(button.dataset.view));
    selectMethod(0);
    renderTraining();
    renderProvenance();
    $("#loading").hidden = true;
    $("#workspace").hidden = false;
  } catch (error) {
    $("#loading").hidden = true;
    $("#load-error").hidden = false;
    $("#load-error").textContent = `无法载入实验回放。${error.message}\n请确认使用 EvidenceGate 的 serve 命令打开已生成的报告；直接双击 HTML 文件无法读取本地 API。`;
  }
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildMetricCards, formatLoss, scenarioName, answerRiskStatus, decisionCard, trainingRuns, trainingEvidence };
} else {
  init();
}
