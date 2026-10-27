"""Independent public-data audit fixtures, without model imports or generation."""
import copy
import importlib.util
from pathlib import Path

import pytest

from evidence_gate.data import decision, messages_for

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_v3_data', ROOT / 'scripts/audit_v3_data.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
VARIANTS = [f'{semantic}_b{budget}' for semantic in ('supported', 'expired', 'ambiguous') for budget in (0, 1)]


def fixture_rows():
    rows = []
    for split_index, split in enumerate(('train', 'dev', 'test')):
        for index in range(4):
            number = split_index * 4 + index
            product, other = f'FIXTURE-{number}', f'NOISE-{number + 1000}'
            related, equal = bool(index // 2), bool(index % 2)
            def doc(id_, entity, plan, value, source='official', expired=False):
                return {'id': id_, 'source': source, 'valid_from': '2026-01-01',
                        'valid_until': '2026-04-01' if expired else '2026-12-31',
                        'text': f"On {entity}'s {plan} plan, audit logs are retained for {value} days."
                                + (' ' + auditor.NOISE if source == 'community' else '')}
            for semantic in ('supported', 'expired', 'ambiguous'):
                for budget in (0, 1):
                    documents = [doc('doc-1', product, 'Starter', 30, expired=semantic == 'expired'),
                                 doc('doc-2', product, 'Team', 30 if equal else 60),
                                 doc('doc-3', product if related else other, 'Starter', 999, 'community')]
                    question = f'For {product} on the Starter plan, how many days is the audit-log retention period?'
                    if semantic == 'ambiguous':
                        question = f'For {product}, how many days is the audit-log retention period?'
                    observation = {'question': question, 'as_of': '2026-04-01', 'budget': budget, 'documents': documents}
                    action = 'answer' if semantic == 'supported' else ('clarify' if semantic == 'ambiguous' else ('search' if budget else 'abstain'))
                    target = decision('answer', '30 days', ['doc-1']) if action == 'answer' else decision(action)
                    group = f'{split}-group-{number}'
                    variant = f'{semantic}_b{budget}'
                    rows.append({'id': f'{group}:{variant}', 'group_id': group, 'variant': variant,
                                 'split': split, 'observation': observation, 'messages': messages_for(observation),
                                 'target': target, 'gold': {'product': product,
                                   'catalogue': [doc('recovery-1', product, 'Starter', 30), doc('recovery-2', product, 'Team', 60)],
                                   'recovery_target': decision('answer', '30 days', ['recovery-1'])}})
    return rows


def resync(row):
    row['messages'] = messages_for(row['observation'])


def test_audit_uses_public_labels_paired_budgets_and_both_cue_counterexamples():
    rows = fixture_rows()
    before = copy.deepcopy(rows)
    result = auditor.audit_v3(rows, expected_variants=VARIANTS)
    assert result['rows'] == result['public_rule_agreements'] == result['recovery_agreements'] == 72
    assert result['budget_paired_contexts'] == 36
    assert result['queried_product_count'] == 12
    assert result['model_predictions_read'] is False and result['llm_calls'] == 0
    assert result['previously_seen_entity_overlap'] == 0
    assert result['fixed_shortcut_probes']['dev']['v2_text_shortcut']['nonanswer_false_answer'] > 0
    assert result['train_fitted_lookup_probes']['budget']['test']['rows'] == 24
    assert rows == before


@pytest.mark.parametrize('mutation,match', [
    ('label', 'Public contract'), ('messages', 'boundary'), ('pair', 'both budget'),
    ('incomplete', 'complete V3 variant'), ('catalogue', 'catalogue disagree'),
    ('seen', 'previously seen'), ('same_product_other_group', 'crosses product'),
    ('wrong_recovery_target', 'Declared recovery'),
])
def test_inconsistent_or_leaking_data_is_rejected(mutation, match):
    rows, seen = fixture_rows(), []
    if mutation == 'label':
        rows[0]['target'] = decision('abstain')
    elif mutation == 'messages':
        rows[0]['messages'][1]['content'] += ' target=answer'
    elif mutation == 'pair':
        rows[1]['observation']['documents'][1]['valid_until'] = '2027-01-01'
        resync(rows[1])
    elif mutation == 'incomplete':
        rows.pop()
    elif mutation == 'catalogue':
        rows[0]['gold']['catalogue'][0]['text'] = rows[0]['gold']['catalogue'][0]['text'].replace('30 days', '90 days')
    elif mutation == 'seen':
        seen = [{'unrelated_metadata': rows[0]['gold']['product']}]
    elif mutation == 'same_product_other_group':
        # Even an entity only present in another group's private catalogue is checked.
        rows[-1]['gold']['foreign_reference'] = rows[0]['gold']['product']
    elif mutation == 'wrong_recovery_target':
        rows[0]['gold']['recovery_target']['value'] = '90 days'
    with pytest.raises(ValueError, match=match):
        auditor.audit_v3(rows, seen, VARIANTS)


@pytest.mark.parametrize('feature', ['all_product', 'equal_value'])
def test_one_sided_perfect_answer_cue_is_rejected(feature):
    rows = fixture_rows()
    for row in rows:
        if row['variant'].startswith('expired'):
            if feature == 'all_product':
                # A wrong-product community doc does not change the task label.
                product = row['gold']['product']
                for doc in row['observation']['documents']:
                    if doc['source'] == 'community':
                        doc['text'] = doc['text'].replace(product, 'AUXILIARY-' + product.split('-')[-1])
            else:
                row['observation']['documents'][1]['text'] = row['observation']['documents'][1]['text'].replace('30 days', '60 days')
            resync(row)
    with pytest.raises(ValueError, match='Missing positive/negative counterexamples'):
        auditor.audit_v3(rows, expected_variants=VARIANTS)


def test_features_use_exact_product_entities_not_substring_matching():
    observation = fixture_rows()[0]['observation']
    observation['documents'][2]['text'] = observation['documents'][2]['text'].replace('NOISE-1000', 'FIXTURE-00')
    assert auditor.public_features(observation)['all_documents_mention_product'] is False
