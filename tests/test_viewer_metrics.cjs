const assert = require("node:assert/strict");
const test = require("node:test");
const { buildMetricCards } = require("../src/evidence_gate/web/app.js");
const find = (cards, label) => cards.find(card => card.label === label);

test("zero parseable answers cannot be displayed as zero unsafe-answer risk", () => {
  const cards = buildMetricCards({ n: 280, groups: 20, format_valid_rate: 0,
    unsupported_answer_rate: null, denominators: { emitted_answers: 0, answerable: 80 } });
  const risk = find(cards, "无依据作答比例");
  assert.equal(risk.value, "不可估计");
  assert.match(risk.note, /没有可解析的 answer.*格式失败不能视为安全/);
  assert.equal(find(cards, "格式有效率").value, "0.0%");
});

test("overall success does not conceal answerable or complete-group failure", () => {
  const cards = buildMetricCards({ n: 280, groups: 20, success_rate: 200 / 280,
    answerable_success: 0.25, all_variants_correct_rate: 0, action_accuracy: 200 / 280,
    format_valid_rate: 1, unsupported_answer_rate: 0,
    denominators: { emitted_answers: 20, answerable: 80 } });
  assert.equal(cards.length, 6);
  assert.equal(find(cards, "整体成功率").value, "71.4%");
  assert.equal(find(cards, "可答成功率").value, "25.0%");
  assert.equal(find(cards, "整组全通过率").value, "0.0%");
  assert.equal(find(cards, "无依据作答比例").value, "0.0%");
  assert.match(find(cards, "无依据作答比例").note, /20 次可解析 answer 为分母/);
  assert.match(find(cards, "可答成功率").note, /80 个本应作答/);
});

test("unsafe-answer proportion uses emitted-answer denominator, not all examples", () => {
  const risk = find(buildMetricCards({ n: 280, format_valid_rate: 0.5,
    unsupported_answer_rate: 0.75, denominators: { emitted_answers: 40 } }), "无依据作答比例");
  assert.equal(risk.value, "75.0%");
  assert.match(risk.note, /40 次可解析 answer 为分母/);
  assert.match(risk.note, /50.0%/);
});

test("missing denominator is unknown rather than a fabricated zero", () => {
  const risk = find(buildMetricCards({ unsupported_answer_rate: 0 }), "无依据作答比例");
  assert.equal(risk.value, "—");
  assert.match(risk.note, /未记录/);
  assert.equal(find(buildMetricCards(), "整体成功率").value, "—");
});
