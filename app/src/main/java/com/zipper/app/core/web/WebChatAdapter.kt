package com.zipper.app.core.web

import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import com.zipper.app.R

class WebChatAdapter(
    private val messages: MutableList<WebChatMessage> = mutableListOf(),
    private val onOpenWeb: (String) -> Unit = {},
    private val onOpenApp: (String) -> Unit = {},
    private val onFeedbackSuccess: (WebChatMessage) -> Unit = {},
    private val onFeedbackFail: (WebChatMessage) -> Unit = {},
    private val onReadAloud: (String) -> Unit = {}
) : RecyclerView.Adapter<WebChatAdapter.ChatViewHolder>() {

    fun addMessage(msg: WebChatMessage) {
        messages.add(msg)
        notifyItemInserted(messages.size - 1)
    }

    fun updateLastProgress(newText: String) {
        val lastIdx = messages.indexOfLast { it.type == ChatMessageType.PROGRESS_STEP }
        if (lastIdx != -1) {
            messages[lastIdx] = messages[lastIdx].copy(text = newText)
            notifyItemChanged(lastIdx)
        } else {
            addMessage(WebChatMessage(type = ChatMessageType.PROGRESS_STEP, text = newText))
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChatViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_web_chat_message, parent, false)
        return ChatViewHolder(view)
    }

    override fun onBindViewHolder(holder: ChatViewHolder, position: Int) {
        holder.bind(messages[position])
    }

    override fun getItemCount(): Int = messages.size

    inner class ChatViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val llUserContainer: LinearLayout = itemView.findViewById(R.id.llUserContainer)
        private val tvUserMessage: TextView = itemView.findViewById(R.id.tvUserMessage)

        private val llProgressContainer: LinearLayout = itemView.findViewById(R.id.llProgressContainer)
        private val tvProgressText: TextView = itemView.findViewById(R.id.tvProgressText)

        private val llAssistantContainer: LinearLayout = itemView.findViewById(R.id.llAssistantContainer)
        private val tvAssistantMessage: TextView = itemView.findViewById(R.id.tvAssistantMessage)
        private val btnReadAloud: TextView = itemView.findViewById(R.id.btnReadAloud)
        private val cvResultCard: MaterialCardView = itemView.findViewById(R.id.cvResultCard)
        private val tvResultTitle: TextView = itemView.findViewById(R.id.tvResultTitle)
        private val tvResultPrice: TextView = itemView.findViewById(R.id.tvResultPrice)
        private val tvResultSnippet: TextView = itemView.findViewById(R.id.tvResultSnippet)
        private val tvResultRating: TextView? = itemView.findViewById(R.id.tvResultRating)
        private val tvResultStore: TextView? = itemView.findViewById(R.id.tvResultStore)
        private val hsvImageGallery: android.widget.HorizontalScrollView? = itemView.findViewById(R.id.hsvImageGallery)
        private val llImageGallery: LinearLayout? = itemView.findViewById(R.id.llImageGallery)
        private val llComparisonOffers: LinearLayout = itemView.findViewById(R.id.llComparisonOffers)
        private val llSingleActions: LinearLayout = itemView.findViewById(R.id.llSingleActions)
        private val btnOpenWeb: Button = itemView.findViewById(R.id.btnOpenWeb)
        private val btnOpenApp: Button = itemView.findViewById(R.id.btnOpenApp)

        private val llFeedbackSection: LinearLayout = itemView.findViewById(R.id.llFeedbackSection)
        private val tvFeedbackPrompt: TextView = itemView.findViewById(R.id.tvFeedbackPrompt)
        private val llFeedbackActions: LinearLayout = itemView.findViewById(R.id.llFeedbackActions)
        private val btnConfirmSuccess: Button = itemView.findViewById(R.id.btnConfirmSuccess)
        private val btnConfirmFail: Button = itemView.findViewById(R.id.btnConfirmFail)
        private val tvFeedbackStatus: TextView = itemView.findViewById(R.id.tvFeedbackStatus)

        fun bind(msg: WebChatMessage) {
            llUserContainer.visibility = View.GONE
            llProgressContainer.visibility = View.GONE
            llAssistantContainer.visibility = View.GONE
            cvResultCard.visibility = View.GONE

            when (msg.type) {
                ChatMessageType.USER -> {
                    llUserContainer.visibility = View.VISIBLE
                    tvUserMessage.text = msg.text
                }
                ChatMessageType.PROGRESS_STEP -> {
                    llProgressContainer.visibility = View.VISIBLE
                    tvProgressText.text = msg.text
                }
                ChatMessageType.ASSISTANT -> {
                    llAssistantContainer.visibility = View.VISIBLE
                    tvAssistantMessage.text = msg.text
                    btnReadAloud.setOnClickListener {
                        onReadAloud(msg.text)
                    }

                    msg.resultData?.let { data ->
                        cvResultCard.visibility = View.VISIBLE
                        tvResultTitle.text = data.title
                        if (!data.price.isNullOrBlank()) {
                            tvResultPrice.visibility = View.VISIBLE
                            tvResultPrice.text = data.price
                        } else {
                            tvResultPrice.visibility = View.GONE
                        }
                        if (!data.rating.isNullOrBlank()) {
                            tvResultRating?.visibility = View.VISIBLE
                            tvResultRating?.text = data.rating
                        } else {
                            tvResultRating?.visibility = View.GONE
                        }
                        val storeName = data.storeName ?: WebDeepLinkManager.resolveStoreName(data.webUrl, data.appPackage)
                        if (storeName.isNotBlank() && storeName != "Store") {
                            tvResultStore?.visibility = View.VISIBLE
                            tvResultStore?.text = storeName
                            btnOpenApp.text = "🛒 Open in $storeName"
                        } else {
                            tvResultStore?.visibility = View.GONE
                            btnOpenApp.text = "🛒 Open App"
                        }
                        if (!data.snippet.isNullOrBlank()) {
                            tvResultSnippet.visibility = View.VISIBLE
                            tvResultSnippet.text = data.snippet
                        } else {
                            tvResultSnippet.visibility = View.GONE
                        }

                        // Extracted 2-3 Visual Images Rendering
                        if (data.imageUrls.isNotEmpty() && hsvImageGallery != null && llImageGallery != null) {
                            hsvImageGallery.visibility = View.VISIBLE
                            llImageGallery.removeAllViews()
                            val ctx = itemView.context
                            val cardSize = (110 * ctx.resources.displayMetrics.density).toInt()
                            val margin = (8 * ctx.resources.displayMetrics.density).toInt()
                            val radius = (10 * ctx.resources.displayMetrics.density)

                            for (imgUrl in data.imageUrls.take(3)) {
                                val card = MaterialCardView(ctx).apply {
                                    this.radius = radius
                                    cardElevation = 0f
                                    setCardBackgroundColor(android.graphics.Color.parseColor("#131D33"))
                                    strokeWidth = (1 * ctx.resources.displayMetrics.density).toInt()
                                    setStrokeColor(android.graphics.Color.parseColor("#334155"))
                                    val lp = LinearLayout.LayoutParams(cardSize, cardSize)
                                    lp.setMargins(0, 0, margin, 0)
                                    layoutParams = lp
                                }

                                val iv = android.widget.ImageView(ctx).apply {
                                    layoutParams = ViewGroup.LayoutParams(
                                        ViewGroup.LayoutParams.MATCH_PARENT,
                                        ViewGroup.LayoutParams.MATCH_PARENT
                                    )
                                    scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
                                }
                                card.addView(iv)
                                llImageGallery.addView(card)

                                WebImageLoader.load(ctx, imgUrl, iv)
                            }
                        } else {
                            hsvImageGallery?.visibility = View.GONE
                        }

                        // Multi-Store Comparison Offers rendering
                        if (data.comparisonOffers.isNotEmpty()) {
                            llSingleActions.visibility = View.GONE
                            llComparisonOffers.visibility = View.VISIBLE
                            llComparisonOffers.removeAllViews()

                            val ctx = itemView.context
                            for (offer in data.comparisonOffers) {
                                val row = LinearLayout(ctx).apply {
                                    orientation = LinearLayout.VERTICAL
                                    setBackgroundColor(android.graphics.Color.parseColor("#131D33"))
                                    val pad = (8 * ctx.resources.displayMetrics.density).toInt()
                                    setPadding(pad, pad, pad, pad)
                                    val lp = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.MATCH_PARENT,
                                        LinearLayout.LayoutParams.WRAP_CONTENT
                                    )
                                    lp.setMargins(0, (4 * ctx.resources.displayMetrics.density).toInt(), 0, (4 * ctx.resources.displayMetrics.density).toInt())
                                    layoutParams = lp
                                }

                                val topRow = LinearLayout(ctx).apply {
                                    orientation = LinearLayout.HORIZONTAL
                                    gravity = android.view.Gravity.CENTER_VERTICAL
                                    layoutParams = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.MATCH_PARENT,
                                        LinearLayout.LayoutParams.WRAP_CONTENT
                                    )
                                }

                                val storeColor = when (offer.storeName.lowercase()) {
                                    "amazon" -> "#F59E0B"
                                    "flipkart" -> "#2563EB"
                                    "zepto" -> "#84CC16"
                                    "blinkit" -> "#F43F5E"
                                    else -> "#06B6D4"
                                }

                                val tvStore = TextView(ctx).apply {
                                    text = offer.storeName.uppercase()
                                    setTextColor(android.graphics.Color.parseColor(storeColor))
                                    textSize = 11f
                                    paint.isFakeBoldText = true
                                }

                                val tvPrice = TextView(ctx).apply {
                                    text = "  ${offer.priceDisplay}"
                                    setTextColor(android.graphics.Color.parseColor("#34D399"))
                                    textSize = 14f
                                    paint.isFakeBoldText = true
                                }

                                topRow.addView(tvStore)
                                topRow.addView(tvPrice)

                                if (offer.isBestDeal) {
                                    val tvDeal = TextView(ctx).apply {
                                        text = "  🏆 Lowest"
                                        setTextColor(android.graphics.Color.parseColor("#FBBF24"))
                                        textSize = 10f
                                        paint.isFakeBoldText = true
                                    }
                                    topRow.addView(tvDeal)
                                }

                                row.addView(topRow)

                                val tvItemTitle = TextView(ctx).apply {
                                    text = offer.title
                                    setTextColor(android.graphics.Color.parseColor("#CBD5E1"))
                                    textSize = 11f
                                    maxLines = 2
                                    ellipsize = android.text.TextUtils.TruncateAt.END
                                    val lpTitle = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.MATCH_PARENT,
                                        LinearLayout.LayoutParams.WRAP_CONTENT
                                    )
                                    lpTitle.setMargins(0, 2, 0, 8)
                                    layoutParams = lpTitle
                                }
                                row.addView(tvItemTitle)

                                val btnStore = Button(ctx).apply {
                                    text = if (offer.webUrl.startsWith("zipper://")) "🛒 Order on ${offer.storeName}" else "🛒 Open in ${offer.storeName}"
                                    textSize = 11f
                                    setTextColor(android.graphics.Color.WHITE)
                                    backgroundTintList = android.content.res.ColorStateList.valueOf(
                                        if (offer.isBestDeal) android.graphics.Color.parseColor("#059669")
                                        else android.graphics.Color.parseColor("#1E293B")
                                    )
                                    layoutParams = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.MATCH_PARENT,
                                        (36 * ctx.resources.displayMetrics.density).toInt()
                                    )
                                    setOnClickListener {
                                        if (offer.webUrl.startsWith("zipper://")) {
                                            onOpenWeb(offer.webUrl)
                                        } else {
                                            val handled = WebDeepLinkManager.openDeepLink(ctx, offer.webUrl, offer.appPackage)
                                            if (!handled) {
                                                offer.appPackage?.let { p -> onOpenApp(p) }
                                            }
                                        }
                                    }
                                }
                                row.addView(btnStore)

                                llComparisonOffers.addView(row)
                            }
                        } else {
                            llComparisonOffers.visibility = View.GONE
                            llSingleActions.visibility = View.VISIBLE

                            btnOpenWeb.setOnClickListener {
                                data.webUrl?.let { url -> onOpenWeb(url) }
                            }
                            btnOpenApp.setOnClickListener {
                                val url = data.webUrl.orEmpty()
                                val pkg = data.appPackage
                                val handled = WebDeepLinkManager.openDeepLink(itemView.context, url, pkg)
                                if (!handled) {
                                    data.appPackage?.let { p -> onOpenApp(p) }
                                }
                            }
                        }

                        llFeedbackSection.visibility = View.VISIBLE
                        if (data.isFeedbackSubmitted) {
                            llFeedbackActions.visibility = View.GONE
                            tvFeedbackStatus.visibility = View.VISIBLE
                            tvFeedbackStatus.text = data.feedbackMessage ?: "Feedback recorded."
                        } else {
                            llFeedbackActions.visibility = View.VISIBLE
                            tvFeedbackStatus.visibility = View.GONE

                            btnConfirmSuccess.setOnClickListener {
                                data.isFeedbackSubmitted = true
                                data.feedbackMessage = "🎉 Verified as Succeeded! Saved as 1-Tap Quick Replay."
                                llFeedbackActions.visibility = View.GONE
                                tvFeedbackStatus.visibility = View.VISIBLE
                                tvFeedbackStatus.text = data.feedbackMessage
                                onFeedbackSuccess(msg)
                            }

                            btnConfirmFail.setOnClickListener {
                                data.isFeedbackSubmitted = true
                                data.feedbackMessage = "🔄 Marked as Incomplete. Retrying task..."
                                llFeedbackActions.visibility = View.GONE
                                tvFeedbackStatus.visibility = View.VISIBLE
                                tvFeedbackStatus.text = data.feedbackMessage
                                onFeedbackFail(msg)
                            }
                        }
                    }
                }
            }
        }
    }
}
