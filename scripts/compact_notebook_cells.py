"""Presentation-only source transformation for the expanded baseline notebook."""
import ast


def compact_baseline_cell(text):
    if 'with compact_output(' in text:
        return text
    tree = ast.parse(text)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {'collect_shadow_paper_features', 'collect_target_paper_features'}):
            calls.append(node)
    if not calls:
        return text
    filename = ('pfv_scoring.log' if 'MISSING_MASKS' in text else
                'baseline_shadow_scoring.log' if 'collect_shadow_paper_features(' in text
                else 'baseline_target_scoring.log')
    lines = text.splitlines(keepends=True)
    for node in sorted(calls, key=lambda node: node.lineno, reverse=True):
        indent = ' ' * node.col_offset
        block = lines[node.lineno - 1:node.end_lineno]
        lines[node.lineno - 1:node.end_lineno] = [
            f"{indent}with compact_output(RUN_DIR / 'logs' / '{filename}'):\n",
            *['    ' + line for line in block],
        ]
    text = ('from src.notebook_output import compact_output\n'
            f"print('Detailed progress and warnings:', RUN_DIR / 'logs' / '{filename}')\n"
            + ''.join(lines))
    text = text.replace('display(pd.DataFrame(shadow_query_diagnostics).T)',
        "pd.DataFrame(shadow_query_diagnostics).T.to_json(RUN_DIR / 'logs' / 'shadow_query_diagnostics.json', indent=2)\n"
        "display(pd.DataFrame(shadow_query_diagnostics).T[['record_queries', 'walk_censored_fraction']])")
    text = text.replace('display(pd.DataFrame(attack_result[\'metrics\']).T)',
        "display(pd.DataFrame(attack_result['metrics']).T[['accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc']].round(4))")
    text = text.replace('display(pd.Series(target_query_diagnostics, name=\'expanded target query diagnostics\'))',
        "display(pd.Series({key: target_query_diagnostics[key] for key in ['record_queries', 'walk_censored_fraction']}, name='target query diagnostics'))")
    text = text.replace('display(missing_comparison)',
        "display(missing_comparison[['n_missing', 'accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc', 'auc_change_from_complete']].round(4))")
    text = text.replace("(MISSING_DIR / 'protocol.json').write_text", "_ = (MISSING_DIR / 'protocol.json').write_text")
    ast.parse(text)
    return text
