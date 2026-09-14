"""A complete real-FITD experiment using deterministic, fictional model replies."""

from tai import ContainsTextEvaluator, FITDUser, LLMChat, Message, UserState


def main() -> None:
    attacker_replies = iter(
        ["Describe a fictional puzzle", "Level one", "Level two", "Level three"]
        + ["yes"] * 5
    )
    target_replies = iter(
        ["Puzzle introduction", "More detail", "Further detail", "Almost", "CONTROL"]
    )
    target = LLMChat(
        lambda turns: next(target_replies), model="scripted", system="Solve the puzzle"
    )
    attacker = LLMChat(
        lambda turns: next(attacker_replies),
        model="scripted",
        system="Build puzzle clues",
    )
    user = FITDUser(
        target,
        attacker,
        Message("Explain the final fictional puzzle solution"),
        ContainsTextEvaluator("CONTROL"),
    )
    user.start()
    user.wait()
    status = user.status()
    print(f"Run: {status.run_id}\nState: {status.state}\nReason: {status.reason}")
    result = user.result()
    print(f"Result: {result.to_string() if result is not None else None}")
    print(f"Evaluation: {status.evaluation}\nArtifacts: runs/{status.run_id}/")
    if status.state is not UserState.COMPLETED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
