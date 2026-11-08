"""Decode raw UniWeaver model responses into canonical stage schemas."""
import hashlib
import json
import re

from .contract import find_planner_placeholders
from .policy_registry import (
    JSON_DUPLICATE_KEY_POLICY,
    JSON_MISORDERED_CLOSURE_POLICY,
    JSON_MISSING_COMMA_POLICY,
    JSON_PAYLOAD_EXTRACTION_POLICY,
    JSON_SPLIT_OBJECT_POLICY,
    JSON_SYNTAX_CLOSURE_POLICY,
    JSON_TRAILING_COMMA_POLICY,
)


def split_think(raw_text):
    """Split a model response into its reasoning block and following answer."""
    raw_text = raw_text or ''
    index = raw_text.find('</think>')
    if index == -1:
        index = raw_text.find('</think')
    if index != -1:
        close_length = 8 if raw_text[index:index + 8] == '</think>' else 7
        after = raw_text[index + close_length:].strip()
        while after.startswith(('</think>', '</think')):
            after = after[8 if after.startswith('</think>') else 7:].strip()
        return raw_text[:index + close_length], after
    return raw_text, raw_text.strip()


def detect_repetition_loop(text, min_repeat=4, min_chunk=80, label='Output'):
    """Describe the pattern an exactly-repeating response is stuck on."""
    text = text or ''
    if len(text) < min_chunk * min_repeat:
        return None
    for window in (300, 200, 100, 60):
        if len(text) < window * min_repeat:
            continue
        # A repeated body can be followed by a unique terminator, and a window
        # anchored on the very last character would carry that terminator with
        # it and match nothing. Step the window back before giving up on it.
        reach = min(len(text) - window * min_repeat, 96)
        for offset in range(0, reach + 1, 16):
            end = len(text) - offset
            chunk = text[end - window:end]
            count = text.count(chunk)
            if count >= min_repeat:
                return f'{label} loop: {window}-char pattern repeated {count}x'
    return None


def detect_think_loop(think_text, min_repeat=4, min_chunk=80):
    """Describe a repeated reasoning pattern, or return ``None``."""
    return detect_repetition_loop(
        think_text, min_repeat=min_repeat, min_chunk=min_chunk, label='Think')


def _required_keys_present(parsed, required_keys):
    return (not required_keys or
            isinstance(parsed, dict) and all(key in parsed for key in required_keys))


def _repair_common_typo_outside_strings(text):
    """Repair the known bracket typo without changing quoted JSON values."""
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if text.startswith('}}]}', index):
            output.append('}]}')
            index += 4
            continue
        output.append(char)
        index += 1
    return ''.join(output)


def _insert_missing_comma(candidate, error):
    """Insert the comma the parser says is missing, or return ``None``."""
    if not isinstance(error, json.JSONDecodeError):
        return None
    if error.msg != "Expecting ',' delimiter":
        return None
    position = error.pos
    if position <= 0 or position >= len(candidate):
        return None
    if not candidate[position:].strip():
        return None
    repaired = candidate[:position] + ',' + candidate[position:]
    try:
        json.loads(repaired)
    except (TypeError, ValueError):
        return None
    return repaired


def _remove_trailing_commas_outside_strings(text):
    """Remove only a comma whose next non-space token is a closing delimiter."""
    output = []
    index = 0
    in_string = False
    escaped = False
    changed = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ',':
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in ']}':
                changed = True
                index += 1
                continue
        output.append(char)
        index += 1
    return ''.join(output), changed


def _unambiguous_delimiter_closure(text):
    """Return a closed JSON suffix only for one stack-determined completion."""
    start_positions = [position for position in (text.find('{'), text.find('['))
                       if position >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = text[start:].rstrip('`').rstrip()
    stack = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in '{[':
            stack.append('}' if char == '{' else ']')
        elif char in '}]':
            if not stack or char != stack[-1]:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    return candidate + ''.join(reversed(stack))


def _misordered_delimiter_closure(text):
    """Return ``(candidate, offset, expected, found)`` for one mis-ordered closer."""
    start_positions = [position for position in (text.find('{'), text.find('['))
                       if position >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = text[start:].rstrip('`').rstrip()
    stack: list[str] = []
    in_string = False
    escaped = False
    dropped = None
    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in '{[':
            stack.append('}' if char == '{' else ']')
        elif char in '}]':
            if stack and char == stack[-1]:
                stack.pop()
                continue
            if dropped is not None or candidate[index + 1:].strip():
                return None
            dropped = (index, stack[-1] if stack else '', char)
    if dropped is None or in_string or not stack:
        return None
    offset, expected, found = dropped
    repaired = candidate[:offset] + candidate[offset + 1:]
    return repaired + ''.join(reversed(stack)), offset, expected, found


def _sha256(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def _operation(policy_id, operation, before, after, **details):
    result = {
        'policy_id': policy_id,
        'operation': operation,
        'input_sha256': _sha256(before),
        'output_sha256': _sha256(after),
    }
    result.update(details)
    return result


def _loads_with_duplicate_keys(candidate):
    duplicates = []

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                duplicates.append({
                    'key': key,
                    'previous_value_sha256': _sha256(json.dumps(
                        value[key], ensure_ascii=False, sort_keys=True)),
                    'last_value_sha256': _sha256(json.dumps(
                        item, ensure_ascii=False, sort_keys=True)),
                    'same_value': value[key] == item,
                })
            # JSON object semantics select the final occurrence.  We expose
            # that selection rather than pretending the duplicate was absent.
            value[key] = item
        return value

    return json.loads(candidate, object_pairs_hook=object_pairs), duplicates


def _json_candidates(text):
    """Yield distinct likely JSON payloads from one normalized response."""
    seen = set()

    def add(candidate, method):
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            return candidate, method
        return None

    direct = add(text, 'direct')
    if direct:
        yield direct

    for part in text.split('```'):
        part = part.strip()
        if part.startswith('json'):
            part = part[len('json'):].strip()
        fenced = add(part, 'markdown')
        if fenced:
            yield fenced

    first_brace = text.find('{')
    if first_brace >= 0:
        prefixed = add(text[first_brace:], 'text_prefix')
        if prefixed:
            yield prefixed

        depth = 0
        in_string = False
        escaped = False
        block_start = None
        complete_blocks = []
        for index, char in enumerate(text[first_brace:], first_brace):
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == '{':
                if depth == 0:
                    block_start = index
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and block_start is not None:
                    complete_blocks.append(text[block_start:index + 1])
                    block_start = None

        for block in sorted(complete_blocks, key=len, reverse=True):
            matched = add(block, 'brace_match')
            if matched:
                yield matched

        if depth > 0 and block_start is not None and not in_string:
            repaired = _unambiguous_delimiter_closure(text[block_start:])
            if repaired:
                repaired = add(repaired, 'stack_closed')
                if repaired:
                    yield repaired

        # Brace counting alone can read a mis-ordered tail as a complete block
        # (``...,"text":"B"}}`` closes the object brace twice), so this repair is
        # attempted on the prefix regardless of what the brace scan concluded.
        misordered = _misordered_delimiter_closure(text[first_brace:])
        if misordered:
            misordered = add(misordered[0], 'misordered_closed')
            if misordered:
                yield misordered

    first_bracket = text.find('[')
    if first_bracket >= 0:
        array = add(text[first_bracket:], 'array')
        if array:
            yield array


def parse_json_with_metadata(raw, required_keys=None):
    """Parse one response with immutable, ordered recovery evidence."""
    if not raw or not str(raw).strip():
        return None, 'empty_output', []
    source = str(raw)
    text = source.strip()
    operations = []
    normalized = False
    think_end = text.rfind('</think>')
    if think_end >= 0:
        text = text[think_end + len('</think>'):].strip()
        normalized = True
    if text.startswith('###Assistant:'):
        text = text[len('###Assistant:'):].lstrip()
        normalized = True
    elif text.startswith('###Assistant'):
        text = text[len('###Assistant'):].lstrip(':').lstrip()
        normalized = True
    stripped = text.lstrip('\ufeff\u200b\u200c\u200d\u200e\u200f\ufffe> ').strip()
    normalized = normalized or stripped != text
    text = stripped
    if normalized:
        operations.append(_operation(
            JSON_PAYLOAD_EXTRACTION_POLICY['policy_id'],
            'normalize_transport_wrapper',
            source, text))
    repaired_typo = _repair_common_typo_outside_strings(text)
    if repaired_typo != text:
        operations.append(_operation(
            JSON_SYNTAX_CLOSURE_POLICY['policy_id'],
            'repair_known_bracket_typo',
            text, repaired_typo))
    repaired_text, trailing_comma_removed = _remove_trailing_commas_outside_strings(
        repaired_typo)
    if trailing_comma_removed:
        operations.append(_operation(
            JSON_TRAILING_COMMA_POLICY['policy_id'],
            'remove_trailing_comma',
            repaired_typo, repaired_text))

    missing_keys_candidate = None
    # A schema-valid block built from the prompt's own example values is an echo;
    # hold it aside so a later block with real content wins.
    placeholder_candidate = None
    for candidate, method in _json_candidates(repaired_text):
        candidate_operations = list(operations)
        if candidate != repaired_text:
            if method == 'stack_closed':
                policy = JSON_SYNTAX_CLOSURE_POLICY
                operation = 'close_delimiter_stack'
                details = {}
            elif method == 'misordered_closed':
                policy = JSON_MISORDERED_CLOSURE_POLICY
                operation = 'drop_misordered_closer_then_close_stack'
                misordered = _misordered_delimiter_closure(repaired_text)
                prefix_starts = [position for position in (
                    repaired_text.find('{'), repaired_text.find('['))
                    if position >= 0]
                details = (
                    {'dropped_offset': misordered[1],
                     'expected_closer': misordered[2],
                     'found_closer': misordered[3],
                     'json_prefix_offset': min(prefix_starts)}
                    if misordered and prefix_starts else {}
                )
            else:
                policy = JSON_PAYLOAD_EXTRACTION_POLICY
                operation = method
                details = {}
            candidate_operations.append(_operation(
                policy['policy_id'], operation, repaired_text, candidate,
                **details))
        try:
            parsed, duplicates = _loads_with_duplicate_keys(candidate)
        except (TypeError, ValueError) as error:
            # One missing comma between members is recoverable punctuation, not
            # a semantic guess: retry the same candidate with just that comma.
            comma_repaired = _insert_missing_comma(candidate, error)
            if comma_repaired is None:
                continue
            try:
                parsed, duplicates = _loads_with_duplicate_keys(comma_repaired)
            except (TypeError, ValueError):
                continue
            candidate_operations.append(_operation(
                JSON_MISSING_COMMA_POLICY['policy_id'],
                'insert_missing_comma',
                candidate, comma_repaired,
                offset=error.pos, line=error.lineno, column=error.colno))
            candidate = comma_repaired
            method = 'missing_comma_repaired'
        if duplicates:
            canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            for duplicate in duplicates:
                operation = _operation(
                    JSON_DUPLICATE_KEY_POLICY['policy_id'],
                    'select_last_duplicate_key_value', candidate, canonical,
                    **duplicate)
                operation['input_sha256'] = duplicate['previous_value_sha256']
                operation['output_sha256'] = duplicate['last_value_sha256']
                candidate_operations.append(operation)
        if _required_keys_present(parsed, required_keys):
            if repaired_typo != text and method == 'direct':
                method = 'common_typo_repaired'
            elif trailing_comma_removed and method == 'direct':
                method = 'trailing_comma_repaired'
            elif normalized and method == 'direct':
                method = 'normalized_direct'
            if find_planner_placeholders(parsed):
                if placeholder_candidate is None:
                    placeholder_candidate = (parsed, method, candidate_operations)
                continue
            return parsed, method, candidate_operations
        if missing_keys_candidate is None:
            missing_keys_candidate = parsed

    merged = {}
    merge_conflict = False
    for match in re.finditer(r'\{[\s\S]*?\}', repaired_text):
        try:
            parsed, _ = _loads_with_duplicate_keys(match[0])
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            if set(merged) & set(parsed):
                merge_conflict = True
                break
            merged.update(parsed)
    if (not merge_conflict and merged
            and _required_keys_present(merged, required_keys)):
        merged_text = json.dumps(merged, ensure_ascii=False, sort_keys=True)
        operations.append(_operation(
            JSON_SPLIT_OBJECT_POLICY['policy_id'],
            'merge_non_overlapping_objects',
            repaired_text, merged_text))
        if not find_planner_placeholders(merged):
            return merged, 'split_objects', operations
        if placeholder_candidate is None:
            placeholder_candidate = (merged, 'split_objects', operations)
    if placeholder_candidate is not None:
        # Validation reports which example was echoed.
        return placeholder_candidate
    if missing_keys_candidate is not None:
        return missing_keys_candidate, 'missing_keys', operations
    if '{' in repaired_text and '}' not in repaired_text[repaired_text.find('{'):]:
        return None, 'json_truncated_no_close', operations
    if '"' in repaired_text and '{' not in repaired_text:
        return None, 'no_json_structure', operations
    return None, 'parse_failed', operations


def _decode(raw, required_keys):
    parsed, parse_method, recovery_operations = parse_json_with_metadata(
        raw, required_keys)
    candidate = parsed
    if parse_method == 'missing_keys':
        parsed = None
    for sequence, operation in enumerate(recovery_operations, 1):
        operation['sequence'] = sequence
    return {
        'parsed': parsed,
        'candidate': candidate,
        'issues': [],
        'parse_method': parse_method,
        'recovery_operations': recovery_operations,
    }


def decode_steps(raw):
    # Field-synonym acceptance belongs to the registered repair
    # (``plan1_step_alias_v1``); a decoding issue would be fatal for the render gate.
    return _decode(raw, ['steps'])


def decode_plan4(raw):
    return _decode(raw, ['reviewer_checklist'])


def read_cached_step_text(generation):
    """Read the exact UniWeaver deterministic Step projection."""
    parsed = generation.get('parsed') if isinstance(generation, dict) else None
    required = ('semantic_image_prompt', 'model_image_prompt', 'content')
    if (not isinstance(parsed, dict)
            or any(not isinstance(parsed.get(key), str)
                   or not parsed[key].strip() for key in required)):
        return None
    return {key: parsed[key] for key in required}


def condition_refs(raw_conditions):
    """Read references without filtering unknown keys."""
    refs = []
    if not isinstance(raw_conditions, dict):
        return refs
    for value in raw_conditions.values():
        if not isinstance(value, list):
            continue
        for entry in value:
            if isinstance(entry, str):
                refs.append(entry)
    return refs


def _reference_list(value, context, issues):
    if not isinstance(value, list):
        issues.append(f'{context} must be a list of condition keys')
        return []

    refs = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            issues.append(f'{context} contains a non-string condition reference')
            continue
        refs.append(entry)
    return refs


def decode_plan3(raw, step_count):
    """Decode only the canonical UniWeaver Plan3 shape."""
    result = _decode(raw, ['condition_bank'])
    plan3 = result['parsed']
    if not isinstance(plan3, dict):
        return result

    issues = result['issues']
    condition_bank = plan3.get('condition_bank')
    if not isinstance(condition_bank, dict):
        issues.append('condition_bank must be an object')
        return result

    raw_items = condition_bank.get('items', [])
    raw_conditions = plan3.get('step_conditions', {})
    result['raw_condition_items'] = raw_items
    result['raw_step_conditions'] = raw_conditions

    if not isinstance(raw_items, list):
        issues.append('condition_bank.items must be a list')
        items = []
    else:
        items = []
        used_keys = set()
        for index, item in enumerate(raw_items, 1):
            if not isinstance(item, dict):
                issues.append(f'condition item {index} must be an object')
                continue
            key = item.get('key')
            if not isinstance(key, str) or not key.strip():
                issues.append(f'condition item {index} has no usable key')
            else:
                if key in used_keys:
                    issues.append(f'duplicate condition key: {key}')
                used_keys.add(key)
            items.append(dict(item))

    if not isinstance(raw_conditions, dict):
        issues.append('step_conditions must be an object')
        conditions = {}
    else:
        conditions = {}
        for step_key, value in raw_conditions.items():
            if step_key in conditions:
                issues.append(f'duplicate step condition entry: {step_key}')
            refs = _reference_list(value, step_key, issues)
            conditions[step_key] = refs

    expected_steps = {f'step_{index}' for index in range(1, step_count + 1)}
    missing_steps = sorted(expected_steps - set(conditions))
    if missing_steps:
        issues.append(f'step_conditions is missing entries: {missing_steps}')

    # Never overwrite the model JSON with the usable subset. Consumers validate
    # and either accept the complete raw block or ignore Plan3 atomically.
    result['condition_items'] = items
    result['step_conditions'] = conditions
    return result
