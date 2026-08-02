# Hazard 1 — Postmortem

Fill this in as you go. Ten minutes. Section 2 goes in **before** you investigate, and it
is the only section that stops working if you write it afterwards.

---

## 1. Symptom

What did you observe? Be literal: what was printed, what was not, how long you waited,
what the CPU was doing.

```
```

---

## 2. First hypothesis

Written **before** investigating. Wrong is fine and expected.

```
```

---

## 3. How you confirmed it

Which tool, which command, what it showed. Paste the useful output.

```
```

Which rank was the culprit, and how did you tell it apart from the ranks that were merely
blocked?

```
```

---

## 4. Root cause

One or two sentences. Name the specific line and the specific mechanism.

```
```

---

## 5. The fix

What you changed, and which of the alternatives you chose if there was more than one.

```
```

Why that one?

```
```

---

## 6. The general lesson

The section that pays back. What class of bug is this, and what habit prevents it?

```
```

---

## 7. How would you catch this earlier?

If this happened on a 512-rank job at 3am, what would you want to have already been in
place? Think about what `TORCH_NCCL_TRACE_BUFFER_SIZE` and `monitored_barrier` are for.

```
```

---

## 8. Time spent

Hypothesis to fix: `____` minutes.

Worth recording. The fifth hazard should take much less than the first, and watching that
number fall is the actual measure of whether this lab worked.
