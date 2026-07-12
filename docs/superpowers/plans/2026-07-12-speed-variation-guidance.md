# Speed Variation Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add practical speed-variation and reversal guidance to the push-box teleoperator checklist.

**Architecture:** Extend the existing episode-quality bullets rather than creating a duplicate section, and add one dataset-bias warning under “Don't.”

**Tech Stack:** Markdown

## Global Constraints

- Change only `docs/teleoperator_guidelines.md` after this plan.
- Preserve the three existing speed levels and 5 Hz control semantics.
- Leave the untracked `tmp/` directory untouched.

---

### Task 1: Expand teleoperator speed guidance

**Files:**
- Modify: `docs/teleoperator_guidelines.md:69-75`

**Interfaces:**
- Consumes: the existing “What good episodes contain” and “Don't” checklists.
- Produces: explicit operator policy for speed variation, persistence, coverage, and safe reversal.

- [ ] **Step 1: Update episode-quality guidance**

Replace the single all-speeds bullet with guidance that says:

```markdown
- Use all three speed levels during both contact and contact-free movement, and
  vary speed both within individual episodes and across the session.
- Hold a selected speed for several 5 Hz ticks rather than changing it every
  tick; cover each speed across workspace regions, approach directions, and
  both contact and contact-free motion.
- Before reversing a 10 mm command, release the arrows for a zero-action tick
  or step down to speed 1 to avoid an abrupt full-speed reversal.
```

- [ ] **Step 2: Add the dataset-bias warning**

Add under “Don't”:

```markdown
- Don't consistently use fast motion only near the base and slow motion only
  while extended; cover speeds across configurations so speed is not confounded
  with arm position.
```

- [ ] **Step 3: Verify Markdown and commit**

Run:

```bash
git diff --check
git diff -- docs/teleoperator_guidelines.md
```

Expected: no whitespace errors and only the approved checklist additions.

Commit:

```bash
git add docs/teleoperator_guidelines.md
git commit -m "docs: clarify speed variation during collection"
```
