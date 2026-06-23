"""Verify qwq:32b determinism and compare current-code vs April-code test results."""
import json
import sys
from pathlib import Path

def check_batch(path: str, label: str):
    p = Path(path)
    if not p.exists():
        print(f"[{label}] File not found: {path}")
        return
    records = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    n = len(records)
    asr = sum(1 for r in records if r.get("attack_success")) / n * 100 if n else 0
    print(f"\n[{label}] N={n}, ASR={asr:.0f}%")

    # Check byte-determinism: are all runs identical in agent_logs content?
    def extract_content(record):
        return [str(l.get("content", "")) for l in record.get("agent_logs", [])]

    contents = [extract_content(r) for r in records]
    all_identical = all(c == contents[0] for c in contents[1:])
    print(f"  Byte-identical across runs: {all_identical}")

    if not all_identical:
        # Find first diverging run
        for i, c in enumerate(contents[1:], 1):
            if c != contents[0]:
                # Find first diverging entry
                for j, (a, b) in enumerate(zip(contents[0], c)):
                    if a != b:
                        print(f"  First divergence: run {i}, entry {j}")
                        print(f"    Run 0: ...{a[max(0,len(a)//2-40):len(a)//2+40]}...")
                        print(f"    Run {i}: ...{b[max(0,len(b)//2-40):len(b)//2+40]}...")
                        break
                break

    # Show S2 entry[4] for first run (the critical divergence point vs April)
    s2_reasoning = [l for l in records[0].get("agent_logs", [])
                    if l.get("session_index") == 2 and l.get("type") == "reasoning"]
    if len(s2_reasoning) > 1:
        text = s2_reasoning[1].get("content", "")
        print(f"  S2 reasoning[1] char 648 context: ...{text[620:700]}...")


if __name__ == "__main__":
    print("=" * 70)
    print("QWQ:32B DETERMINISM & CODE-ISOLATION VERIFICATION")
    print("=" * 70)

    check_batch("results/qwq_ollama_version_test/results.jsonl", "Current code (N=10)")
    check_batch("results/qwq_april_code_test/results.jsonl", "April code (N=10)")

    # Compare the two if both exist
    curr = Path("results/qwq_ollama_version_test/results.jsonl")
    april = Path("results/qwq_april_code_test/results.jsonl")
    if curr.exists() and april.exists():
        cr = json.loads(curr.read_text().splitlines()[0])
        ar = json.loads(april.read_text().splitlines()[0])

        # Compare S2 entry[4] between current-code and april-code
        def get_s2_4(record):
            s2 = [l for l in record.get("agent_logs", []) if l.get("session_index") == 2]
            return s2[4].get("content", "") if len(s2) > 4 else ""

        c_text = get_s2_4(cr)
        a_text = get_s2_4(ar)
        if c_text == a_text:
            print("\n[COMPARISON] Current-code S2[4] == April-code S2[4]: IDENTICAL")
            print("  → Code change is NOT the cause. Environment confirmed.")
        else:
            for j, (ca, ct) in enumerate(zip(c_text, a_text)):
                if ca != ct:
                    print(f"\n[COMPARISON] S2[4] diverges at char {j}")
                    print(f"  Current: ...{c_text[max(0,j-30):j+50]}...")
                    print(f"  April:   ...{a_text[max(0,j-30):j+50]}...")
                    break
            print("  → Investigate whether code or environment caused the difference.")
