"""Small, explicit programming curriculum; no LLM decides prerequisite policy."""

from models.enums import DifficultyLevel

CURRICULUM = {
    "Python Basics": {
        "aliases": ("python basics", "python基础", "python 基础", "基础语法"),
        "prerequisites": (),
        "summary": "Names refer to values; lists are ordered, mutable collections.",
        "example": "values = [1, 2, 3]; print(sum(values))  # 6",
        "mcq": "Which Python collection is ordered and mutable?",
        "options": ["list", "tuple", "frozenset", "int"],
        "numerical": "What is len([10, 20, 30])?",
        "answer": "3",
    },
    "Recursion": {
        "aliases": ("recursion", "recursive", "递归"),
        "prerequisites": ("Python Basics",),
        "summary": "A recursive function needs a base case and progress toward it.",
        "example": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "mcq": "What prevents unbounded recursion?",
        "options": [
            "a reachable base case",
            "another recursive call",
            "a global list",
            "printing n",
        ],
        "numerical": "What is factorial(4)?",
        "answer": "24",
    },
    "Linked Lists": {
        "aliases": ("linked list", "linked lists", "链表"),
        "prerequisites": ("Python Basics",),
        "summary": "Each node stores a value and a reference to the next node.",
        "example": "while node is not None:\n    print(node.value)\n    node = node.next",
        "mcq": "What does a singly linked list node use to reach its successor?",
        "options": [
            "a next reference",
            "a fixed array index",
            "a recursion limit",
            "a hash of all nodes",
        ],
        "numerical": "How many nodes are in 1 -> 2 -> 3 -> None?",
        "answer": "3",
    },
    "Binary Trees": {
        "aliases": ("binary tree", "binary trees", "二叉树"),
        "prerequisites": ("Recursion",),
        "summary": "Preorder visits root-left-right; inorder visits left-root-right.",
        "example": "def inorder(node):\n    if node:\n        inorder(node.left)\n        print(node.value)\n        inorder(node.right)",
        "mcq": "Which traversal order is inorder?",
        "options": ["left-root-right", "root-left-right", "right-root-left", "root-right-left"],
        "numerical": "A root has two leaf children. How many nodes are there?",
        "answer": "3",
    },
    "DFS": {
        "aliases": ("depth first search", "depth-first search", "dfs", "深度优先"),
        "prerequisites": ("Recursion",),
        "summary": "DFS explores a branch before siblings; track visited vertices in a graph.",
        "example": "def dfs(v):\n    if v in seen: return\n    seen.add(v)\n    for w in graph[v]: dfs(w)",
        "mcq": "What prevents repeatedly visiting a cycle during graph DFS?",
        "options": ["a visited set", "an empty queue", "sorting every edge", "a random seed"],
        "numerical": "A path graph A-B-C has how many reachable vertices from A?",
        "answer": "3",
    },
    "Dynamic Programming": {
        "aliases": ("dynamic programming", "动态规划", " dp"),
        "prerequisites": ("Recursion",),
        "summary": "Reuse solutions to overlapping subproblems with a clear state and transition.",
        "example": "a, b = 0, 1\nfor _ in range(n):\n    a, b = b, a + b  # a becomes Fibonacci(n)",
        "mcq": "Why store previously solved subproblems?",
        "options": [
            "to avoid repeated computation",
            "to remove all base cases",
            "to randomise transitions",
            "to increase recursion depth",
        ],
        "numerical": "With F(0)=0 and F(1)=1, what is F(6)?",
        "answer": "8",
    },
}


def canonical_topic(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    for title, data in CURRICULUM.items():
        if text == title.lower():
            return title
        if any(alias in text for alias in data["aliases"]):
            return title
    return None


def difficulty_for(score: float | None) -> DifficultyLevel:
    if score is None or score < 0.5:
        return DifficultyLevel.BEGINNER
    return DifficultyLevel.INTERMEDIATE if score < 0.8 else DifficultyLevel.ADVANCED


def recommend_next(scores: dict[str, float]) -> dict:
    """Prioritise measured weakness, then unmet prerequisites, then unseen topics."""
    known = {canonical_topic(key): value for key, value in scores.items() if canonical_topic(key)}
    weak = [(score, title) for title, score in known.items() if score < 0.6]
    if weak:
        _, target = min(weak)
        cause = f"mastery of {target} is {known[target]:.0%}, below 60%"
    else:
        target = next((title for title in CURRICULUM if title not in known), "Dynamic Programming")
        cause = (
            f"{target} has not been assessed"
            if target not in known
            else "all measured concepts meet the review threshold"
        )
    chosen = target
    while True:
        missing = next(
            (item for item in CURRICULUM[chosen]["prerequisites"] if known.get(item, 0) < 0.6),
            None,
        )
        if missing is None:
            break
        chosen = missing
    rationale = cause
    if chosen != target:
        rationale += f"; study prerequisite {chosen} first (mastery {known.get(chosen, 0):.0%})"
    return {
        "next_topic": chosen,
        "target_topic": target,
        "rationale": rationale,
        "difficulty": difficulty_for(known.get(chosen)).value,
    }
