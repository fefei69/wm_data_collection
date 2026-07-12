# Speed Variation Guidance Design

## Goal

Make the teleoperator guidelines explicit about how to vary the three movement
speeds without creating abrupt commands or unwanted correlations in the
training dataset.

## Placement and wording

Expand `docs/teleoperator_guidelines.md` under “What good episodes contain,”
beside the existing instruction to use all three speed levels. The guidance
will require variation both within and across episodes, recommend holding each
speed for several 5 Hz ticks, and require coverage across workspace regions and
both contact and contact-free motion.

Add a reversal-safety instruction to use a zero-action tick or speed 1 before
reversing a 10 mm command. Under “Don't,” warn against consistently pairing
fast motion with near-base poses and slow motion with extended poses, since
that would confound speed with robot configuration.

## Verification

Review the rendered Markdown structure, run `git diff --check`, and confirm the
change affects only the intended guideline and this approved design note.
