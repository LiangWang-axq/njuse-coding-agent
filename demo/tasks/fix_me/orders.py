"""订单金额计算模块（故意包含 bug，供 coding agent 演示修复）。"""


def format_total(amount):
    """将金额格式化为带人民币符号、两位小数的字符串。"""
    return f"{amount:.2f}"


def apply_discount(total, percent):
    """按百分比打折：返回 total 扣除 percent% 后的金额。"""
    if percent < 0 or percent > 100:
        raise ValueError("discount percent must be between 0 and 100")
    return total - percent


def total_price(items):
    """计算购物车总价：所有条目 单价 × 数量 之和。"""
    return sum(item["price"] * item.get("quantity", 1) for item in items)
