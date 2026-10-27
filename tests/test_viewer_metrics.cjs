const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { buildMetricCards, scenarioName, answerRiskStatus, decisionCard, trainingRuns, trainingEvidence, formatLoss } = require("../src/evidence_gate/web/app.js");
const find = (cards, label) => cards.find(card => card.label === label);
const report = JSON.parse(fs.readFileSync(path.join(__dirname, "../experiments/v2/report.json"), "utf8"));

test("zero parseable answers cannot be displayed as zero unsafe-answer risk", () => {
  const cards = buildMetricCards({ n: 280, groups: 20, format_valid_rate: 0,
    unsupported_answer_rate: null, denominators: { emitted_answers: 0, answerable: 80 } });
  const risk = find(cards, "证据合同未通过的回答比例");
  assert.equal(risk.value, "不可估计");
  assert.match(risk.note, /没有可解析的 answer.*格式失败不能视为安全/);
  assert.equal(find(cards, "格式有效率").value, "0.0%");
});

test("overall success does not conceal answerable or complete-group failure", () => {
  const cards = buildMetricCards({ n: 280, groups: 20, success_rate: 200 / 280,
    answerable_success: 0.25, all_variants_correct_rate: 0, action_accuracy: 200 / 280,
    format_valid_rate: 1, unsupported_answer_rate: 0,
    denominators: { emitted_answers: 20, answerable: 80 } });
  assert.equal(cards.length, 8);
  assert.equal(find(cards, "整体成功率").value, "71.4%");
  assert.equal(find(cards, "可答成功率").value, "25.0%");
  assert.equal(find(cards, "整组全通过率").value, "0.0%");
  assert.equal(find(cards, "证据合同未通过的回答比例").value, "0.0%");
  assert.match(find(cards, "证据合同未通过的回答比例").note, /20 次可解析 answer 为分母/);
  assert.match(find(cards, "可答成功率").note, /80 个本应作答/);
});

test("unsafe-answer proportion uses emitted-answer denominator, not all examples", () => {
  const risk = find(buildMetricCards({ n: 280, format_valid_rate: 0.5,
    unsupported_answer_rate: 0.75, denominators: { emitted_answers: 40 } }), "证据合同未通过的回答比例");
  assert.equal(risk.value, "75.0%");
  assert.match(risk.note, /40 次可解析 answer 为分母/);
  assert.match(risk.note, /50.0%/);
});

test("missing denominator is unknown rather than a fabricated zero", () => {
  const risk = find(buildMetricCards({ unsupported_answer_rate: 0 }), "证据合同未通过的回答比例");
  assert.equal(risk.value, "—");
  assert.match(risk.note, /未记录/);
  assert.equal(find(buildMetricCards(), "整体成功率").value, "—");
});

test("real V2 report preserves the risk denominators and exposes both decision metrics", () => {
  const selected = report.methods.find(method => method.name === "selected-v2");
  const cards = buildMetricCards(selected.summary);
  assert.equal(find(cards, "可答成功率").value, "12.5%");
  assert.equal(find(cards, "整组全通过率").value, "0.0%");
  assert.equal(find(cards, "动作 Macro-F1").value, "71.2%");
  assert.match(find(cards, "动作 Macro-F1").note, /四种动作.*等权/);
  assert.equal(find(cards, "不应作答却作答比例").value, "3.0%");
  assert.match(find(cards, "不应作答却作答比例").note, /200 个本不应作答/);
  const risk = find(cards, "证据合同未通过的回答比例");
  assert.equal(risk.value, "37.5%");
  assert.match(risk.note, /16 次可解析 answer/);
  assert.match(risk.note, /不应回答时回答、错值、错引或缺引/);
  const base = report.methods.find(method => method.name === "base");
  assert.equal(find(buildMetricCards(base.summary), "证据合同未通过的回答比例").value, "不可估计");
});

test("all frozen 24-cell variant labels retain semantic names and exact budgets", () => {
  // Read protocol labels only; this test does not load new held-out examples or predictions.
  const protocol = JSON.parse(fs.readFileSync(path.join(__dirname, "../experiments/v31/protocol.json"), "utf8"));
  const labels = protocol.variants.map(scenarioName);
  assert.equal(labels.length, 24);
  assert.equal(new Set(labels).size, 24);
  for (const [i, variant] of protocol.variants.entries()) {
    assert.notEqual(labels[i], variant);
    assert.match(labels[i], new RegExp(`查询预算 ${variant.slice(-1)}$`));
  }
  assert.match(scenarioName("consensus_b0"), /多份官方证据一致/);
  assert.match(scenarioName("irrelevant_topic_b1"), /无关主题干扰/);
  assert.equal(scenarioName("missing_search"), "缺少依据 · 可查询");
  assert.equal(scenarioName("unknown_b1"), "unknown_b1");
});

test("citation failures are not relabelled as fabricated facts or as safety", () => {
  // Minimal display fixture, not a generated model result or a saved V3.1 report.
  const display = answerRiskStatus({ prediction: { action: "answer" },
    metrics: { format_valid: true, citation_correct: false, unsupported_answer: true } });
  assert.equal(display.kind, "fail");
  assert.match(display.label, /引用集合错误/);
  assert.doesNotMatch(display.label, /幻觉|事实错误/);
  const invalid = answerRiskStatus({ prediction: null, metrics: { format_valid: false, unsupported_answer: false } });
  assert.notEqual(invalid.kind, "pass");
  assert.match(invalid.label, /无法估计/);
});

test("decision cards render every citation as literal text, including two-source consensus", () => {
  class FakeNode {
    constructor(tag = "text") { this.tag = tag; this.children = []; this.attributes = {}; this.ownText = ""; }
    append(...nodes) { this.children.push(...nodes); }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener() {}
    set textContent(value) { this.ownText = String(value); }
    get textContent() { return this.ownText + this.children.map(node => node.textContent).join(""); }
  }
  const previousDocument = global.document, previousNode = global.Node;
  try {
    global.Node = FakeNode;
    global.document = { createElement: tag => new FakeNode(tag), createTextNode: text => {
      const node = new FakeNode(); node.textContent = text; return node;
    } };
    const card = decisionCard({ action: "answer", value: "30 days", citations: ["official-A", "<script>official-B</script>"] }, "unit fixture", true, "model");
    const nodes = node => [node, ...node.children.flatMap(nodes)];
    const chips = nodes(card).filter(node => node.className === "citation-chip");
    assert.deepEqual(chips.map(node => node.textContent), ["official-A", "<script>official-B</script>"]);
    assert.match(card.textContent, /引用（2）/);
    assert.equal(nodes(card).filter(node => node.tag === "script").length, 0);
  } finally { global.document = previousDocument; global.Node = previousNode; }
});

test("real training metadata is visible without treating record count as total budget", () => {
  const runs = trainingRuns(report);
  const selected = runs.find(run => run.name === "selected-v2-step-600");
  const shown = trainingEvidence(selected);
  assert.equal(find(shown.fields, "所选检查点").value, "step 600");
  assert.equal(find(shown.fields, "本轮已完成训练").value, "600 steps");
  assert.equal(find(shown.fields, "本轮训练总预算").value, "600 steps");
  assert.match(shown.note, /训练前缀/);
  const older = trainingEvidence(runs.find(run => run.name === "answer-only"));
  assert.equal(find(older.fields, "本轮已完成训练").value, "200 steps");
  assert.equal(find(older.fields, "本轮训练总预算").value, "未记录");
});

test("an early selected prefix stays distinct from a completed larger budget", () => {
  // Deliberately tiny display fixture; no experimental scores are fabricated.
  const shown = trainingEvidence({ records: [{ step: 2, loss: 0.2 }],
    metadata: { selected_step: 2, optimizer_steps: 2, completed_training_steps: 8, total_training_budget_steps: 8 } });
  assert.equal(find(shown.fields, "所选检查点").value, "step 2");
  assert.equal(find(shown.fields, "本轮已完成训练").value, "8 steps");
  assert.equal(find(shown.fields, "本轮训练总预算").value, "8 steps");
  const missing = trainingEvidence({ records: [{ step: 2, loss: 0.2 }], metadata: {} });
  assert.ok(missing.fields.every(field => field.value === "未记录"));
  assert.match(trainingEvidence({ records: [{ step: 8, loss: 0.1 }], metadata: { selected_step: 2 } }).note, /记录不一致/);
});

test("small nonzero training losses retain meaningful precision", () => {
  const run = report.training.find(entry => entry.name === "answer-only");
  assert.equal(formatLoss(run.history.at(-1).loss), "4.508e-5");
  assert.equal(formatLoss(0), "0");
  assert.equal(formatLoss(null), "未记录");
});

test("actual dev navigation preserves focus and list nodes while showing 24 siblings and four methods", () => {
  // An in-memory DOM check of saved dev evaluations, not a fabricated test report.
  const methods = [600, 1200, 1800, 2400].map(step => JSON.parse(fs.readFileSync(
    path.join(__dirname, `../experiments/v31/dev-step-${String(step).padStart(6, "0")}.json`), "utf8")));
  let document;
  class TestNode {
    constructor(tag = "text") { this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {}; this.ownText = ""; this.className = ""; this.replacements = 0; this.scrollTop = 0; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.replacements++; this.children = nodes; }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    addEventListener(key, listener) { this.listeners[key] = listener; }
    focus() { document.activeElement = this; }
    set textContent(value) { this.ownText = String(value); }
    get textContent() { return this.ownText + this.children.map(node => node.textContent).join(""); }
    get classList() { return {
      add: value => { this.className = [...new Set([...this.className.split(" "), value])].join(" "); },
      remove: value => { this.className = this.className.split(" ").filter(item => item !== value).join(" "); },
    }; }
  }
  const elements = new Map();
  document = {
    activeElement: null,
    createElement: tag => new TestNode(tag),
    createTextNode: text => { const node = new TestNode(); node.textContent = text; return node; },
    querySelector: selector => {
      if (selector === "#case-list .case-row.selected") return elements.get("#case-list").children.find(node => node.className.split(" ").includes("selected"));
      if (!elements.has(selector)) elements.set(selector, new TestNode(selector));
      return elements.get(selector);
    },
  };
  const context = vm.createContext({ module: { exports: {} }, document, Node: TestNode, methods });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/evidence_gate/web/app.js"), "utf8"), context);
  vm.runInContext("state.report = { methods }; selectMethod(3)", context);
  const caseList = elements.get("#case-list");
  assert.equal(caseList.children.length, 576);
  assert.equal(elements.get("#scenario-select").children.length, 25);
  const retained = caseList.children.slice(), replacements = caseList.replacements;
  const clicked = retained.at(-1);
  caseList.scrollTop = 9000;
  clicked.focus();
  clicked.listeners.click({ currentTarget: clicked });
  assert.equal(caseList.replacements, replacements);
  assert.ok(caseList.children.every((node, index) => node === retained[index]));
  assert.equal(document.activeElement, clicked);
  assert.equal(caseList.scrollTop, 9000);
  assert.equal(clicked.attributes["aria-pressed"], "true");
  assert.equal(retained[0].attributes["aria-pressed"], "false");
  assert.equal(vm.runInContext("state.selected.id", context), methods[3].cases.at(-1).id);
  const nodes = node => [node, ...node.children.flatMap(nodes)];
  let tables = nodes(elements.get("#case-detail")).filter(node => node.tag === "table");
  assert.deepEqual(tables.map(table => table.children.find(node => node.tag === "tbody").children.length), [24, 4]);
  vm.runInContext("state.scenario = 'consensus_b1'; renderCases()", context);
  assert.equal(caseList.children.length, 24);
  const chips = nodes(elements.get("#case-detail")).filter(node => node.className === "citation-chip");
  const expected = vm.runInContext("state.selected.expected.citations", context);
  const prediction = vm.runInContext("state.selected.prediction.citations", context);
  assert.equal(expected.length, 2);
  assert.deepEqual(chips.map(node => node.textContent), [...expected, ...prediction]);
  tables = nodes(elements.get("#case-detail")).filter(node => node.tag === "table");
  assert.equal(tables[0].children.find(node => node.tag === "tbody").children.length, 24);
});
