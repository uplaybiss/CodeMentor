"""Simplified Chinese display names; internal curriculum IDs stay unchanged."""

import re

from config.curriculum import canonical_topic

TOPIC_LABELS = {
    "Python Basics": "Python 基础",
    "Recursion": "递归",
    "Linked Lists": "链表",
    "Binary Trees": "二叉树",
    "DFS": "深度优先搜索",
    "Dynamic Programming": "动态规划",
}
TOPIC_DESCRIPTIONS = {
    "Python Basics": "从变量、列表和基础表达式开始。",
    "Recursion": "理解终止条件，拆解递归调用。",
    "Linked Lists": "用节点和引用连接一组数据。",
    "Binary Trees": "掌握节点结构与常见遍历顺序。",
    "DFS": "沿一条分支探索，学会避免重复访问。",
    "Dynamic Programming": "定义状态，复用已经求解的子问题。",
}
DIFFICULTY_LABELS = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}
LEVEL_LABELS = {"beginner": "初学者", "intermediate": "进阶学习者", "advanced": "高级学习者"}
MEMORY_LABELS = {
    "profile": "学习档案",
    "preference": "学习偏好",
    "concept": "学习记录",
    "strength": "已知优势",
    "mistake": "易错记录",
    "conversation_summary": "学习摘要",
    "recommendation": "学习建议",
}


def topic_label(value: str | None) -> str:
    if not value:
        return "尚未选择"
    canonical = canonical_topic(value)
    return TOPIC_LABELS.get(canonical, value)


def display_memory(content: str) -> str:
    """Hide retrieval tags and translate known legacy demo facts without rewriting data."""
    content = re.sub(r"\[topic:[^\]]+\]\s*", "", content)
    legacy = re.fullmatch(r"Student engaged with the topic (.+)\.", content)
    if legacy:
        return f"学习了{topic_label(legacy.group(1))}。"
    return content


def policy_rationale(policy: dict, scores: dict[str, float]) -> str:
    target, chosen = policy["target_topic"], policy["next_topic"]
    target_label, chosen_label = topic_label(target), topic_label(chosen)
    if target in scores and scores[target] < 0.6:
        reason = f"{target_label}的当前掌握度为{scores[target]:.0%}，低于60%的复习阈值。"
    elif target not in scores:
        reason = f"{target_label}尚未完成测验，建议按学习顺序建立基础。"
    else:
        reason = "已测验的知识点均达到复习阈值，建议通过应用练习继续巩固。"
    if target != chosen:
        level = f"掌握度为{scores[chosen]:.0%}" if chosen in scores else "尚未完成测验"
        reason += f"它依赖{chosen_label}，而你的{chosen_label}{level}，所以先补齐这个基础。"
    return reason


def recommendation_text(rec: dict | None, scores: list[dict] | None = None) -> str:
    """Present current or known legacy rule output in Chinese; preserve its selected topic."""
    if not rec:
        return "完成一次学习或测验后，这里会给出适合你的下一步建议。"
    rationale = rec.get("rationale", "")
    if re.search(r"[\u4e00-\u9fff]", rationale):
        return rationale
    weak = re.search(r"mastery of (.+?) is (\d+)%, below 60%", rationale)
    prereq = re.search(r"study prerequisite (.+?) first \(mastery (\d+)%\)", rationale)
    if weak:
        text = f"{topic_label(weak.group(1))}的掌握度为{weak.group(2)}%，建议先复习薄弱部分。"
        if prereq:
            text += f"先修知识{topic_label(prereq.group(1))}还需要巩固，因此优先安排这部分学习。"
        return text
    return f"根据最近的学习记录，建议下一步学习{topic_label(rec.get('next_topic'))}。"
