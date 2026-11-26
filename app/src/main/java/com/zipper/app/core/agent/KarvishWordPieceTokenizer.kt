package com.zipper.app.core.agent

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * High-performance, zero-JNI WordPiece Tokenizer for Karvish V52 (all-MiniLM-L6-v2) on Android.
 *
 * Reads 30,522 vocabulary tokens from `models/vocab.txt`.
 * Provides exact WordPiece subword tokenization with character offset mapping for
 * Phase D Learned Span Pointer argument extraction.
 */
class KarvishWordPieceTokenizer(context: Context) {

    companion object {
        private const val TAG = "KarvishWordPieceTok"
        const val PAD_ID = 0L
        const val UNK_ID = 100L
        const val CLS_ID = 101L
        const val SEP_ID = 102L
        const val MASK_ID = 103L
    }

    data class TokenSpan(
        val id: Long,
        val startChar: Int,
        val endChar: Int
    )

    private val vocabMap = HashMap<String, Int>(32000)
    private val idToToken = arrayOfNulls<String>(30525)

    init {
        val t0 = System.currentTimeMillis()
        try {
            context.assets.open("models/vocab.txt").use { inputStream ->
                BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8)).use { reader ->
                    var lineIdx = 0
                    var line = reader.readLine()
                    while (line != null) {
                        val token = line.trim()
                        if (token.isNotEmpty()) {
                            vocabMap[token] = lineIdx
                            if (lineIdx < idToToken.size) {
                                idToToken[lineIdx] = token
                            }
                        }
                        lineIdx++
                        line = reader.readLine()
                    }
                }
            }
            Log.i(TAG, "Loaded WordPiece vocabulary (${vocabMap.size} tokens) in ${System.currentTimeMillis() - t0}ms")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load WordPiece vocab: ${e.message}", e)
        }
    }

    fun idToToken(id: Long): String? {
        val idx = id.toInt()
        return if (idx in 0 until idToToken.size) idToToken[idx] else null
    }

    fun getTokens(tokenIds: LongArray): List<String> {
        return tokenIds.map { id -> idToToken(id) ?: "" }
    }

    /**
     * Splits text into WordPiece tokens with character start and end offsets relative to original text.
     */
    fun tokenizeWithOffsets(text: String): List<TokenSpan> {
        if (text.isEmpty()) return emptyList()

        val spans = ArrayList<TokenSpan>()
        var i = 0
        val len = text.length

        while (i < len) {
            // Skip whitespace
            while (i < len && text[i].isWhitespace()) {
                i++
            }
            if (i >= len) break

            // Check if punctuation
            val c = text[i]
            if (isPunctuation(c)) {
                val tokenStr = c.toString().lowercase()
                val id = vocabMap[tokenStr]?.toLong() ?: UNK_ID
                spans.add(TokenSpan(id, i, i + 1))
                i++
                continue
            }

            // Word token: letters or digits
            val wordStart = i
            while (i < len && !text[i].isWhitespace() && !isPunctuation(text[i])) {
                i++
            }
            val wordEnd = i
            val word = text.substring(wordStart, wordEnd).lowercase()

            // Subword greedy matching
            var subStart = 0
            val wordLen = word.length
            var isBad = false

            val subSpans = ArrayList<TokenSpan>()
            while (subStart < wordLen) {
                var subEnd = wordLen
                var curSubStr: String? = null
                var curId: Long? = null

                while (subStart < subEnd) {
                    val piece = word.substring(subStart, subEnd)
                    val candidate = if (subStart == 0) piece else "##$piece"
                    val vId = vocabMap[candidate]
                    if (vId != null) {
                        curSubStr = candidate
                        curId = vId.toLong()
                        break
                    }
                    subEnd--
                }

                if (curSubStr == null || curId == null) {
                    isBad = true
                    break
                }

                val charStart = wordStart + subStart
                val charEnd = wordStart + subEnd
                subSpans.add(TokenSpan(curId, charStart, charEnd))
                subStart = subEnd
            }

            if (isBad) {
                spans.add(TokenSpan(UNK_ID, wordStart, wordEnd))
            } else {
                spans.addAll(subSpans)
            }
        }

        return spans
    }

    /**
     * Encodes a goal string into a fixed length of 32 tokens:
     * [CLS] + tokens + [SEP] + [PAD]...
     * Returns: (goal_ids: LongArray(32), goal_mask: FloatArray(32), offsets: List<Pair<Int, Int>>)
     */
    fun encodeGoal(goal: String, maxLen: Int = 32): Triple<LongArray, FloatArray, List<Pair<Int, Int>>> {
        val ids = LongArray(maxLen) { PAD_ID }
        val mask = FloatArray(maxLen) { 0f }
        val offsets = ArrayList<Pair<Int, Int>>(maxLen)

        // Slot 0: [CLS]
        ids[0] = CLS_ID
        mask[0] = 1.0f
        offsets.add(0 to 0)

        val rawSpans = tokenizeWithOffsets(goal)
        val budget = maxLen - 2 // leave room for [CLS] and [SEP]
        val numTokens = minOf(rawSpans.size, budget)

        for (idx in 0 until numTokens) {
            val s = rawSpans[idx]
            ids[idx + 1] = s.id
            mask[idx + 1] = 1.0f
            offsets.add(s.startChar to s.endChar)
        }

        // Add [SEP]
        val sepPos = numTokens + 1
        ids[sepPos] = SEP_ID
        mask[sepPos] = 1.0f
        offsets.add(0 to 0)

        // Pad remainder
        for (idx in (sepPos + 1) until maxLen) {
            offsets.add(0 to 0)
        }

        return Triple(ids, mask, offsets)
    }

    /**
     * Encodes a candidate label into fixed length of 16 tokens:
     * [CLS] + tokens + [SEP] + [PAD]...
     * Returns: (cand_label_ids: LongArray(16), cand_label_mask: FloatArray(16))
     */
    fun encodeLabel(label: String, maxLen: Int = 16): Pair<LongArray, FloatArray> {
        val ids = LongArray(maxLen) { PAD_ID }
        val mask = FloatArray(maxLen) { 0f }

        if (label.isBlank()) {
            return ids to mask
        }

        ids[0] = CLS_ID
        mask[0] = 1.0f

        val rawSpans = tokenizeWithOffsets(label)
        val budget = maxLen - 2
        val numTokens = minOf(rawSpans.size, budget)

        for (idx in 0 until numTokens) {
            ids[idx + 1] = rawSpans[idx].id
            mask[idx + 1] = 1.0f
        }

        val sepPos = numTokens + 1
        ids[sepPos] = SEP_ID
        mask[sepPos] = 1.0f

        return ids to mask
    }

    /**
     * Decodes token IDs back to text.
     */
    fun decode(tokenIds: LongArray, startIdx: Int, endIdx: Int): String {
        val sb = java.lang.StringBuilder()
        for (i in startIdx..endIdx) {
            if (i !in tokenIds.indices) continue
            val id = tokenIds[i].toInt()
            if (id == PAD_ID.toInt() || id == CLS_ID.toInt() || id == SEP_ID.toInt()) continue
            val piece = idToToken.getOrNull(id) ?: continue
            if (piece.startsWith("##")) {
                sb.append(piece.substring(2))
            } else {
                if (sb.isNotEmpty()) sb.append(" ")
                sb.append(piece)
            }
        }
        return sb.toString().trim()
    }

    /**
     * Decodes an arbitrary list of token IDs back to text (used for BIO token sequences).
     */
    fun decodeTokens(tokenIds: List<Long>): String {
        val sb = java.lang.StringBuilder()
        for (idLong in tokenIds) {
            val id = idLong.toInt()
            if (id == PAD_ID.toInt() || id == CLS_ID.toInt() || id == SEP_ID.toInt()) continue
            val piece = idToToken.getOrNull(id) ?: continue
            if (piece.startsWith("##")) {
                sb.append(piece.substring(2))
            } else {
                if (sb.isNotEmpty()) sb.append(" ")
                sb.append(piece)
            }
        }
        return sb.toString().trim()
    }

    private fun isPunctuation(c: Char): Boolean {
        val type = Character.getType(c)
        return when (type) {
            Character.CONNECTOR_PUNCTUATION.toInt(),
            Character.DASH_PUNCTUATION.toInt(),
            Character.START_PUNCTUATION.toInt(),
            Character.END_PUNCTUATION.toInt(),
            Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
            Character.FINAL_QUOTE_PUNCTUATION.toInt(),
            Character.OTHER_PUNCTUATION.toInt() -> true
            else -> "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~".indexOf(c) >= 0
        }
    }
}
