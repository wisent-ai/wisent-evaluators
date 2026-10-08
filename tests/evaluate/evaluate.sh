#!/usr/bin/env bash
# Real test of the wisent-evaluators program through its command line: list
# and show describe the registry, and evaluate reads JSON-line requests and
# answers each one, or refuses the first it cannot evaluate, naming its line.
#
# The code_tests cases run real Docker: the model's code and the benchmark's
# tests go into a python:slim container with no network, and the limits the
# requests state are this host's own, measured here (its open-file and
# process limits, the memory Docker reports), so the test chooses no number.
# A host whose limit reads "unlimited" stops the test at that measurement.
#
# Every command, how it ended and what it wrote go to the run's report.txt;
# a failed check stops the run (set -e) after naming itself there.
#
# Usage: WISENT_EVALUATORS=target/debug/wisent-evaluators tests/evaluate/evaluate.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${WISENT_EVALUATORS:?set WISENT_EVALUATORS to the program under test, e.g. target/debug/wisent-evaluators}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/evaluate/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused INPUT ARGS...: the command, fed INPUT on standard
# input, must end the way named. An accepted run's answer is its standard
# output; a refused run's answer is everything it wrote.
run() {
  expected=$1
  input=$2
  shift
  shift
  if [ "$expected" = accepted ]; then
    if printf '%s' "$input" | "$BIN" "$@" >"$ROOT/output"; then outcome=accepted; else outcome=refused; fi
  else
    if printf '%s' "$input" | "$BIN" "$@" &>"$ROOT/output"; then outcome=accepted; else outcome=refused; fi
  fi
  out=$(cat "$ROOT/output")
  printf '$ wisent-evaluators %s\ninput: %s\noutcome: %s\noutput: %s\n\n' "$*" "$input" "$outcome" "$out" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}
# verdict REQUEST: evaluate one request and print its verdict.
verdict() {
  run accepted "$1" evaluate
  jq -r .verdict <<<"$out"
}

# The registry describes itself.
run accepted "" list
for name in exact_match f1 choice math code_tests generation user_specified; do
  grep -q "^$name	" <<<"$out" || fail "list does not name $name"
done
echo "ok: list names every evaluator this test uses" >>"$REPORT"
run accepted "" show f1
refused "options.threshold"
run refused "" show no_such_evaluator
refused "no evaluator is named"

# Answers compared as text.
check "exact_match, lenient" "$(verdict '{"evaluator":"exact_match","response":"Paris!","expected":["paris","Paris, France"]}')" truthful
check "exact_match, wrong" "$(verdict '{"evaluator":"exact_match","response":"Lyon","expected":"Paris"}')" untruthful
check "choice, picked by letter" "$(verdict '{"evaluator":"choice","response":"B) Paris","expected":"B"}')" truthful
check "choice, a word starting with the letter is no pick" "$(verdict '{"evaluator":"choice","response":"Absolutely not","expected":"A"}')" untruthful
check "choice, contrastive" "$(verdict '{"evaluator":"choice","response":"","expected":"B","choices":["B) Paris","A) Lyon"]}')" truthful
check "math, LaTeX equivalence" "$(verdict '{"evaluator":"math","response":"so the answer is \\boxed{\\dfrac{a}{b}}","expected":"\\frac{a}{b}"}')" truthful
check "user_specified" "$(verdict '{"evaluator":"user_specified","response":"","expected":"","options":{"truthful":false}}')" untruthful

# Several requests stream one evaluation per line.
run accepted "$(printf '%s\n%s\n' \
  '{"evaluator":"exact_match","response":"a","expected":"a"}' \
  '{"evaluator":"exact_match","response":"a","expected":"b"}')" evaluate
check "one evaluation per request" "$(jq -s -c 'map(.verdict)' <<<"$out")" '["truthful","untruthful"]'

# Refusals name the request line and what it lacks.
run refused '{"evaluator":"f1","response":"a","expected":"a"}' evaluate
refused "decides with options.threshold"
refused "request line"
run refused '{"evaluator":"no_such_evaluator","response":"a","expected":"a"}' evaluate
refused "no evaluator is named"
run refused '{"evaluator":"code_tests","response":"","expected":""}' evaluate
refused "carries no tests"
run refused 'not json' evaluate
refused "is not a request"

# Code run against tests in the Docker sandbox, under this host's limits.
LIMITS=$(jq -nc \
  --argjson files "$(ulimit -n)" \
  --argjson processes "$(ulimit -u)" \
  --argjson memory "$(docker info --format '{{.MemTotal}}')" \
  '{image: "python:slim", interpreter: "python", open_files: $files, processes: $processes, memory_bytes: $memory}')
echo "limits: $LIMITS" >>"$REPORT"
TESTS='from solution import join
assert join("a", "b") == "ab"'
RIGHT='```python
def join(first, second):
    return first + second
```'
WRONG='def join(first, second):
    return second + first'
code_request() {
  jq -nc --arg response "$1" --arg tests "$TESTS" --argjson options "$LIMITS" \
    '{evaluator: "code_tests", response: $response, expected: "", tests: $tests, options: $options}'
}
check "code_tests, passing code" "$(verdict "$(code_request "$RIGHT")")" truthful
run accepted "$(code_request "$WRONG")" evaluate
check "code_tests, failing code" "$(jq -r .verdict <<<"$out")" untruthful
grep -q AssertionError <<<"$(jq -r .meta.stderr <<<"$out")" ||
  fail "the failing run's stderr does not carry the test's AssertionError"
echo "ok: the failing run reports the test's AssertionError" >>"$REPORT"
CONTRAST=$(jq -nc --arg right "$RIGHT" --arg wrong "$WRONG" --arg tests "$TESTS" --argjson options "$LIMITS" \
  '{evaluator: "code_tests", response: "", expected: "", tests: $tests, choices: [$right, $wrong], options: $options}')
check "code_tests, contrastive" "$(verdict "$CONTRAST")" truthful
MISSING=$(jq -c '.options.interpreter = "no-such-interpreter"' <<<"$(code_request "$RIGHT")")
run refused "$MISSING" evaluate
refused "cannot run no-such-interpreter"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
