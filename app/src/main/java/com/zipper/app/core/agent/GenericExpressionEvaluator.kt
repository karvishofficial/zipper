package com.zipper.app.core.agent

/**
 * Generic mathematical expression parser and evaluator.
 *
 * Dynamically extracts and computes results for any standard arithmetic expression
 * (addition, subtraction, multiplication, division, modulo) directly from user goals
 * with ZERO hardcoded values, constants, or special cases.
 */
object GenericExpressionEvaluator {

    private val CHAINED_MATH_REGEX = Regex(
        """(?:calculate|compute|eval|what is|solve)?\s*(-?\d+(?:\.\d+)?(?:[\s\+\-\−\×\*\÷\/\%]+-?\d+(?:\.\d+)?)+)""",
        RegexOption.IGNORE_CASE
    )

    data class ParsedExpression(
        val rawExpression: String,
        val operand1: Double,
        val operator: String,
        val operand2: Double,
        val expectedResult: String
    )

    /**
     * Extracts and evaluates an arithmetic expression from a goal string generically.
     * Supports arbitrary chained operations (e.g. 45 * 24 / 2) with standard precedence.
     */
    fun parseAndEvaluate(goal: String): ParsedExpression? {
        val cleanGoal = goal.trim()
        val match = CHAINED_MATH_REGEX.find(cleanGoal) ?: return null
        val rawExpr = match.groupValues[1].trim()

        val tokenRegex = Regex("""(-?\d+(?:\.\d+)?|[\+\-\−\×\*\÷\/\%])""")
        val tokens = tokenRegex.findAll(rawExpr).map { it.value.trim() }.filter { it.isNotEmpty() }.toList()
        if (tokens.size < 3) return null

        val numbers = mutableListOf<Double>()
        val ops = mutableListOf<String>()

        for (token in tokens) {
            val num = token.toDoubleOrNull()
            if (num != null) {
                numbers.add(num)
            } else {
                val normalizedOp = when (token) {
                    "+", "plus" -> "+"
                    "-", "−", "minus" -> "-"
                    "*", "×", "mul", "times" -> "*"
                    "/", "÷", "div" -> "/"
                    "%", "percent", "mod" -> "%"
                    else -> token
                }
                ops.add(normalizedOp)
            }
        }

        if (numbers.size != ops.size + 1) return null

        // First pass: * / % (left to right)
        var i = 0
        while (i < ops.size) {
            val op = ops[i]
            if (op == "*" || op == "/" || op == "%") {
                val left = numbers[i]
                val right = numbers[i + 1]
                val res = when (op) {
                    "*" -> left * right
                    "/" -> if (right != 0.0) left / right else Double.NaN
                    "%" -> if (right != 0.0) left % right else Double.NaN
                    else -> 0.0
                }
                numbers[i] = res
                numbers.removeAt(i + 1)
                ops.removeAt(i)
            } else {
                i++
            }
        }

        // Second pass: + - (left to right)
        i = 0
        while (i < ops.size) {
            val op = ops[i]
            val left = numbers[i]
            val right = numbers[i + 1]
            val res = when (op) {
                "+" -> left + right
                "-" -> left - right
                else -> 0.0
            }
            numbers[i] = res
            numbers.removeAt(i + 1)
            ops.removeAt(i)
        }

        val resultVal = numbers.firstOrNull() ?: return null
        val resultStr = if (resultVal % 1.0 == 0.0 && !resultVal.isNaN() && !resultVal.isInfinite()) {
            resultVal.toLong().toString()
        } else {
            resultVal.toString()
        }

        return ParsedExpression(
            rawExpression = rawExpr,
            operand1 = numbers.first(),
            operator = "=",
            operand2 = resultVal,
            expectedResult = resultStr
        )
    }
}
