package com.zipper.app.webmind.data

import java.util.Locale

data class OfflineComparisonItem(
    val title: String,
    val priceDisplay: String,
    val price: Double,
    val store: String,
    val deliverySpeed: String,
    val badge: String,
    val imageUrl: String = "",
    val url: String = "",
    val specs: Map<String, String> = emptyMap()
)

data class OfflineComparisonPayload(
    val title: String,
    val headline: String,
    val summaryMarkdown: String,
    val recommendedItem: OfflineComparisonItem,
    val items: List<OfflineComparisonItem>,
    val verdict: String,
    val isOffline: Boolean = true
)

/**
 * On-Device Offline Produce & Tech Knowledge Base for Karvish Webmind.
 * Enables instant multi-platform comparisons with zero network requests and 0 cloud tokens.
 */
object OfflineKnowledgeCatalog {

    fun lookupOffline(query: String): OfflineComparisonPayload? {
        val richPayload = ComprehensiveKnowledgeCatalog.lookup(query)
        if (richPayload != null && richPayload.products.isNotEmpty()) {
            val items = richPayload.products.map { p ->
                OfflineComparisonItem(
                    title = p.title,
                    priceDisplay = if (p.unitPriceDisplay.isNotBlank()) "${p.priceDisplay} (${p.unitPriceDisplay})" else p.priceDisplay,
                    price = p.price,
                    store = p.store,
                    deliverySpeed = p.deliverySpeed,
                    badge = p.badge,
                    imageUrl = p.imageUrl,
                    url = p.productUrl,
                    specs = p.specs
                )
            }
            val rec = items.firstOrNull { it.badge.contains("Lowest", true) || it.badge.contains("Best", true) } ?: items.first()
            return OfflineComparisonPayload(
                title = richPayload.title,
                headline = richPayload.headline,
                summaryMarkdown = "${richPayload.definitiveVerdict}\n\n${richPayload.savingsProofText}",
                recommendedItem = rec,
                items = items,
                verdict = richPayload.definitiveVerdict,
                isOffline = true
            )
        }

        val classified = WebDomainClassifier.classify(query)
        val entity = classified.cleanEntity.lowercase(Locale.ROOT)

        return when {
            entity.contains("tomato") -> getTomatoComparison()
            entity.contains("onion") -> getOnionComparison()
            entity.contains("potato") -> getPotatoComparison()
            entity.contains("milk") -> getMilkComparison()
            entity.contains("egg") -> getEggComparison()
            entity.contains("paneer") -> getPaneerComparison()
            entity.contains("apple") -> getAppleComparison()
            entity.contains("banana") -> getBananaComparison()
            classified.domain == QueryDomain.QUICK_GROCERY -> getGeneralGroceryComparison(classified.cleanEntity)
            entity.contains("iphone") || entity.contains("s24") || entity.contains("mobile") || entity.contains("phone") -> getMobileComparison(entity)
            entity.contains("laptop") || entity.contains("70k") || entity.contains("gaming") -> getLaptopComparison()
            else -> null
        }
    }

    private fun getTomatoComparison(): OfflineComparisonPayload {
        val zepto = OfflineComparisonItem(
            title = "Fresh Hybrid Tomato (500g)",
            priceDisplay = "₹22 (₹44/kg)",
            price = 22.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Fastest Delivery",
            imageUrl = "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=400",
            url = "https://www.zeptonow.com/search?q=tomato",
            specs = mapOf("Weight" to "500g", "Type" to "Farm Hybrid", "Shelf Life" to "3-4 Days")
        )
        val blinkit = OfflineComparisonItem(
            title = "Country Desi Tomato (1kg)",
            priceDisplay = "₹39 (₹39/kg)",
            price = 39.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 12 mins",
            badge = "Lowest Price 🏆",
            imageUrl = "https://images.unsplash.com/photo-1546470427-227c7369a47d?w=400",
            url = "https://blinkit.com/s/?q=tomato",
            specs = mapOf("Weight" to "1 kg", "Type" to "Country Desi", "Flavour" to "Tangy Curries")
        )
        val mandi = OfflineComparisonItem(
            title = "Local APMC Mandi Bulk Baseline",
            priceDisplay = "₹28 – ₹34/kg",
            price = 30.0,
            store = "Mandi Baseline",
            deliverySpeed = "In-Person",
            badge = "Wholesale Index",
            specs = mapOf("Unit" to "1 kg", "Sourcing" to "Direct Farm Gate")
        )

        return OfflineComparisonPayload(
            title = "Tomato Live & Offline Price Comparison",
            headline = "🍅 Tomato: Quick-Commerce & Mandi Benchmark",
            summaryMarkdown = """
                📊 **Tomato Market Comparison (Offline Baseline):**
                - **Blinkit:** **₹39/kg** (Country Desi 1kg) 🏆 *Lowest Rate*
                - **Zepto:** **₹22 for 500g** (₹44/kg, Farm Hybrid) ⚡ *10-min delivery*
                - **Local Mandi Baseline:** **₹28 – ₹34/kg**

                💡 **Verdict:** For large curry prep, **Blinkit (1kg at ₹39)** saves ~₹5/kg. For salads or immediate 10-minute need, **Zepto 500g** offers crisp hybrid firmness.
            """.trimIndent(),
            recommendedItem = blinkit,
            items = listOf(blinkit, zepto, mandi),
            verdict = "Blinkit offers the lowest per-kg price at ₹39/kg. Zepto delivers fastest in 10 minutes."
        )
    }

    private fun getOnionComparison(): OfflineComparisonPayload {
        val blinkit = OfflineComparisonItem(
            title = "Fresh Red Onion (1kg)",
            priceDisplay = "₹42/kg",
            price = 42.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 11 mins",
            badge = "Lowest Price 🏆",
            imageUrl = "https://images.unsplash.com/photo-1618512496248-a07fe83aa8cb?w=400",
            url = "https://blinkit.com/s/?q=onion",
            specs = mapOf("Weight" to "1 kg", "Size" to "Medium-Large", "Origin" to "Nashik")
        )
        val zepto = OfflineComparisonItem(
            title = "Premium Sambhar Onions (500g)",
            priceDisplay = "₹28 (₹56/kg)",
            price = 28.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Fastest 10m",
            imageUrl = "https://images.unsplash.com/photo-1508747703725-719777637510?w=400",
            url = "https://www.zeptonow.com/search?q=onion",
            specs = mapOf("Weight" to "500g", "Type" to "Small Shallots")
        )
        return OfflineComparisonPayload(
            title = "Onion Price Benchmark",
            headline = "🧅 Onion: Quick-Commerce Benchmark",
            summaryMarkdown = """
                📊 **Onion Multi-Store Pricing:**
                - **Blinkit:** **₹42/kg** (Nashik Red) 🏆 *Best Value*
                - **Zepto:** **₹28 for 500g** (₹56/kg, Shallot/Sambhar) ⚡ *10 mins*
                - **Mandi Baseline:** **₹32 – ₹38/kg**
            """.trimIndent(),
            recommendedItem = blinkit,
            items = listOf(blinkit, zepto),
            verdict = "Blinkit delivers standard cooking onions at ₹42/kg."
        )
    }

    private fun getPotatoComparison(): OfflineComparisonPayload {
        val zepto = OfflineComparisonItem(
            title = "New Crop Potato (1kg)",
            priceDisplay = "₹32/kg",
            price = 32.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Lowest Price 🏆",
            imageUrl = "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=400",
            url = "https://www.zeptonow.com/search?q=potato",
            specs = mapOf("Weight" to "1 kg", "Skin" to "Thin Washed")
        )
        val blinkit = OfflineComparisonItem(
            title = "Fresh Jyoti Potato (1kg)",
            priceDisplay = "₹34/kg",
            price = 34.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 13 mins",
            badge = "Fastest",
            url = "https://blinkit.com/s/?q=potato",
            specs = mapOf("Weight" to "1 kg", "Type" to "Jyoti Sugar-Free")
        )
        return OfflineComparisonPayload(
            title = "Potato Price Comparison",
            headline = "🥔 Potato: Quick-Commerce Comparison",
            summaryMarkdown = """
                📊 **Potato Multi-Store Pricing:**
                - **Zepto:** **₹32/kg** (Thin-skin washed) 🏆 *Lowest Rate*
                - **Blinkit:** **₹34/kg** (Jyoti)
                - **Mandi Average:** **₹24 – ₹28/kg**
            """.trimIndent(),
            recommendedItem = zepto,
            items = listOf(zepto, blinkit),
            verdict = "Zepto is ₹2 cheaper per kg and delivers in 10 minutes."
        )
    }

    private fun getMilkComparison(): OfflineComparisonPayload {
        val nandini = OfflineComparisonItem(
            title = "Nandini Toned Milk (500ml)",
            priceDisplay = "₹22",
            price = 22.0,
            store = "Blinkit / Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Most Popular 🏆",
            specs = mapOf("Volume" to "500 ml", "Fat" to "3.0%", "SNF" to "8.5%")
        )
        val amul = OfflineComparisonItem(
            title = "Amul Taaza Homogenised (500ml)",
            priceDisplay = "₹27",
            price = 27.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Toned Cow",
            specs = mapOf("Volume" to "500 ml", "Fat" to "3.0%", "Pouch" to "UHT Shelf-stable")
        )
        return OfflineComparisonPayload(
            title = "Milk Benchmark Comparison",
            headline = "🥛 Fresh Dairy Milk Benchmark",
            summaryMarkdown = """
                📊 **Milk Multi-Brand Pricing (500ml):**
                - **Nandini Toned:** **₹22** (Standard MRP) 🏆
                - **Amul Taaza Toned:** **₹27**
                - **Country Delight Cow:** **₹38**
            """.trimIndent(),
            recommendedItem = nandini,
            items = listOf(nandini, amul),
            verdict = "Nandini Toned is the standard subsidized fresh daily choice at ₹22/500ml."
        )
    }

    private fun getEggComparison(): OfflineComparisonPayload {
        val tray6 = OfflineComparisonItem(
            title = "Classic White Eggs (6 Pack)",
            priceDisplay = "₹48 (₹8/egg)",
            price = 48.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Quick 6-Pack",
            specs = mapOf("Count" to "6 Eggs", "Protein" to "~6g/egg")
        )
        val tray30 = OfflineComparisonItem(
            title = "Poultry Farm Tray (30 Eggs)",
            priceDisplay = "₹198 (₹6.6/egg)",
            price = 198.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 15 mins",
            badge = "Best Value 🏆",
            specs = mapOf("Count" to "30 Eggs", "Per Egg Rate" to "₹6.60")
        )
        return OfflineComparisonPayload(
            title = "Egg Tray Price Shootout",
            headline = "🥚 Fresh Poultry Eggs Benchmark",
            summaryMarkdown = """
                📊 **Egg Tray Pricing Breakdown:**
                - **30-Egg Tray (Blinkit):** **₹198** (₹6.60/egg) 🏆 *Save ₹42*
                - **6-Egg Pack (Zepto):** **₹48** (₹8.00/egg)
            """.trimIndent(),
            recommendedItem = tray30,
            items = listOf(tray30, tray6),
            verdict = "Buying a 30-egg crate reduces egg price to ₹6.60 compared to ₹8.00 on smaller packs."
        )
    }

    private fun getPaneerComparison(): OfflineComparisonPayload {
        val nandini = OfflineComparisonItem(
            title = "Nandini Fresh Malai Paneer (200g)",
            priceDisplay = "₹85",
            price = 85.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Best Value 🏆",
            specs = mapOf("Weight" to "200g", "Type" to "Malai Cube")
        )
        val milky = OfflineComparisonItem(
            title = "Milky Mist Fresh Paneer (200g)",
            priceDisplay = "₹115",
            price = 115.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 12 mins",
            badge = "Premium Soft",
            specs = mapOf("Weight" to "200g", "Texture" to "Ultra Soft")
        )
        return OfflineComparisonPayload(
            title = "Fresh Paneer Comparison",
            headline = "🧀 Fresh Malai Paneer Benchmark",
            summaryMarkdown = """
                📊 **Paneer 200g Comparison:**
                - **Nandini Malai Paneer:** **₹85** 🏆 *Save ₹30*
                - **Milky Mist Paneer:** **₹115**
                - **Amul Malai Paneer:** **₹92**
            """.trimIndent(),
            recommendedItem = nandini,
            items = listOf(nandini, milky),
            verdict = "Nandini Paneer delivers 200g at ₹85 with identical protein content."
        )
    }

    private fun getAppleComparison(): OfflineComparisonPayload {
        val fuji = OfflineComparisonItem(
            title = "Fresh Royal Gala / Fuji Apples (4 pcs)",
            priceDisplay = "₹139 (~₹35/pc)",
            price = 139.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Crisp & Sweet 🏆",
            specs = mapOf("Count" to "4 pcs (~500g)", "Taste" to "Sweet & Crunchy")
        )
        val shimla = OfflineComparisonItem(
            title = "Himachal Shimla Apples (1kg)",
            priceDisplay = "₹180/kg",
            price = 180.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 14 mins",
            badge = "Local Harvest",
            specs = mapOf("Weight" to "1 kg", "Origin" to "Himachal Pradesh")
        )
        return OfflineComparisonPayload(
            title = "Apple Price Comparison",
            headline = "🍎 Apple: Variety & Price Comparison",
            summaryMarkdown = """
                📊 **Apple Pricing:**
                - **Zepto (4 Pack Royal Gala):** **₹139** 🏆
                - **Blinkit (Shimla 1kg):** **₹180/kg**
            """.trimIndent(),
            recommendedItem = fuji,
            items = listOf(fuji, shimla),
            verdict = "Zepto 4-pack offers imported Royal Gala quality at ₹139."
        )
    }

    private fun getBananaComparison(): OfflineComparisonPayload {
        val yelakki = OfflineComparisonItem(
            title = "Yelakki Small Bananas (500g)",
            priceDisplay = "₹45 (6-8 pcs)",
            price = 45.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Naturally Sweet 🏆",
            specs = mapOf("Type" to "Yelakki / Robusta", "Count" to "~7 pcs")
        )
        val robusta = OfflineComparisonItem(
            title = "Robusta Big Bananas (1kg)",
            priceDisplay = "₹48/kg",
            price = 48.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 12 mins",
            badge = "Budget Staple",
            specs = mapOf("Weight" to "1 kg", "Type" to "Robusta Cavendish")
        )
        return OfflineComparisonPayload(
            title = "Banana Variety Benchmark",
            headline = "🍌 Banana: Yelakki vs Robusta",
            summaryMarkdown = """
                📊 **Banana Comparison:**
                - **Yelakki Bananas (Zepto):** **₹45 for 500g** 🏆 *Top flavor*
                - **Robusta Bananas (Blinkit):** **₹48/kg** *Top volume*
            """.trimIndent(),
            recommendedItem = yelakki,
            items = listOf(yelakki, robusta),
            verdict = "Yelakki provides superior aromatic sweetness; Robusta provides max weight."
        )
    }

    private fun getGeneralGroceryComparison(entity: String): OfflineComparisonPayload {
        val item1 = OfflineComparisonItem(
            title = "$entity (Zepto 10-Minute Catalog)",
            priceDisplay = "₹45 – ₹90",
            price = 45.0,
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Fastest Delivery 🏆",
            url = "https://www.zeptonow.com/search?q=" + java.net.URLEncoder.encode(entity, "UTF-8"),
            specs = mapOf("Category" to "Fresh Quick-Commerce", "Sourcing" to "Dark Store Hub")
        )
        val item2 = OfflineComparisonItem(
            title = "$entity (Blinkit Grocery Catalog)",
            priceDisplay = "₹42 – ₹85",
            price = 42.0,
            store = "Blinkit",
            deliverySpeed = "⚡ 12 mins",
            badge = "Lowest Price",
            url = "https://blinkit.com/s/?q=" + java.net.URLEncoder.encode(entity, "UTF-8"),
            specs = mapOf("Category" to "Quick Grocery", "Dispatch" to "Local Hub")
        )
        return OfflineComparisonPayload(
            title = "$entity Quick-Commerce Comparison",
            headline = "🛒 $entity: Multi-Store Comparison",
            summaryMarkdown = """
                📊 **$entity Cross-Store Baseline:**
                - **Blinkit:** **₹42 – ₹85** 🏆 *Lowest baseline rate*
                - **Zepto:** **₹45 – ₹90** ⚡ *Fastest 10m delivery*
                - **Mandi Index:** Available in fresh daily lots
            """.trimIndent(),
            recommendedItem = item1,
            items = listOf(item1, item2),
            verdict = "Both Zepto and Blinkit stock $entity with delivery in 10-15 minutes."
        )
    }

    private fun getMobileComparison(entity: String): OfflineComparisonPayload {
        val isFlagship = entity.contains("iphone") || entity.contains("s24") || entity.contains("s23") || entity.contains("pixel") || entity == "mobile" || entity == "phone"
        return if (isFlagship) {
            val iphone = OfflineComparisonItem(
                title = "Apple iPhone 16 (128 GB)",
                priceDisplay = "₹77,990",
                price = 77990.0,
                store = "Amazon India",
                deliverySpeed = "Tomorrow",
                badge = "Fastest CPU 🏆",
                imageUrl = "https://m.media-amazon.com/images/I/71d7rfSl0wL._SX679_.jpg",
                url = "https://www.amazon.in/s?k=iPhone+16",
                specs = mapOf(
                    "Chipset" to "Apple A18 (3nm)",
                    "Camera" to "48MP Fusion + 12MP Ultra-wide",
                    "Screen" to "6.1\" Super Retina OLED (60Hz)",
                    "Controls" to "Action Button + Camera Control"
                )
            )
            val s24 = OfflineComparisonItem(
                title = "Samsung Galaxy S24 5G (128 GB)",
                priceDisplay = "₹74,999",
                price = 74999.0,
                store = "Flipkart",
                deliverySpeed = "Tomorrow",
                badge = "Best Display & Zoom 🏆",
                imageUrl = "https://m.media-amazon.com/images/I/717Qo4MH97L._SX679_.jpg",
                url = "https://www.flipkart.com/search?q=Samsung+Galaxy+S24",
                specs = mapOf(
                    "Chipset" to "Snapdragon 8 Gen 3 / Exynos 2400",
                    "Camera" to "50MP Main + 10MP 3x Telephoto + 12MP UW",
                    "Screen" to "6.2\" Dynamic AMOLED 2X (1-120Hz LTPO)",
                    "Features" to "Galaxy AI Live Translate & Circle to Search"
                )
            )
            OfflineComparisonPayload(
                title = "iPhone 16 vs Samsung Galaxy S24 Shootout",
                headline = "📱 Flagship Smartphone Shootout (2025/2026)",
                summaryMarkdown = """
                    📊 **iPhone 16 vs Samsung Galaxy S24 Direct Comparison:**
                    - **Price:** Galaxy S24 is **₹3,000 cheaper** (₹74,999 vs ₹77,990).
                    - **Display:** Galaxy S24 wins with **120Hz LTPO AMOLED** (iPhone 16 is capped at 60Hz).
                    - **Cameras:** Galaxy S24 features a dedicated **3x optical telephoto lens**; iPhone 16 has higher video bitrate and Camera Control button.
                    - **Processor:** iPhone 16 A18 leads single-core benchmarks; S24 leads in multitasking thermals.

                    💡 **Verdict:** Choose **Galaxy S24** for 120Hz fluidity and zoom camera. Choose **iPhone 16** for unmatched resale value, iOS 18 ecosystem, and video creation.
                """.trimIndent(),
                recommendedItem = s24,
                items = listOf(s24, iphone),
                verdict = "Galaxy S24 provides higher value with a 120Hz LTPO display and 3x zoom camera for ₹3,000 less."
            )
        } else {
            // Mid-Range Phone shootout
            val nord = OfflineComparisonItem(
                title = "OnePlus Nord CE4 5G (128 GB)",
                priceDisplay = "₹24,999",
                price = 24999.0,
                store = "Amazon India",
                deliverySpeed = "Tomorrow",
                badge = "Fastest Charging 🏆",
                imageUrl = "https://m.media-amazon.com/images/I/61abLrCfF7L._SX679_.jpg",
                url = "https://www.amazon.in/s?k=OnePlus+Nord+CE4",
                specs = mapOf(
                    "Chipset" to "Snapdragon 7 Gen 3",
                    "Charging" to "100W SUPERVOOC (0-100% in 28m)",
                    "Battery" to "5,500 mAh",
                    "Screen" to "6.7\" 120Hz AMOLED"
                )
            )
            val redmi = OfflineComparisonItem(
                title = "Redmi Note 13 Pro 5G (128 GB)",
                priceDisplay = "₹23,999",
                price = 23999.0,
                store = "Flipkart",
                deliverySpeed = "Tomorrow",
                badge = "200MP Camera 🏆",
                imageUrl = "https://m.media-amazon.com/images/I/71XNeka-BRL._SX679_.jpg",
                url = "https://www.flipkart.com/search?q=Redmi+Note+13+Pro",
                specs = mapOf(
                    "Chipset" to "Snapdragon 7s Gen 2",
                    "Camera" to "200MP OIS Ultra-Clear",
                    "Screen" to "1.5K 120Hz Curved AMOLED",
                    "Charging" to "67W Turbo Charge"
                )
            )
            OfflineComparisonPayload(
                title = "Best Smartphones Under 25,000",
                headline = "📱 Mid-Range Shootout: OnePlus Nord CE4 vs Redmi Note 13 Pro",
                summaryMarkdown = """
                    📊 **Top Mid-Range Mobiles Under ₹25,000:**
                    - **OnePlus Nord CE4:** **₹24,999** (Snapdragon 7 Gen 3, 100W, 5500mAh) 🏆 *Performance King*
                    - **Redmi Note 13 Pro:** **₹23,999** (200MP OIS, 1.5K AMOLED) 📸 *Camera King*
                """.trimIndent(),
                recommendedItem = nord,
                items = listOf(nord, redmi),
                verdict = "OnePlus Nord CE4 leads in processor performance and 100W charging; Redmi Note 13 Pro leads in 200MP photography."
            )
        }
    }

    private fun getLaptopComparison(): OfflineComparisonPayload {
        val loq = OfflineComparisonItem(
            title = "Lenovo LOQ 15 (Core i5 13th Gen, RTX 4050)",
            priceDisplay = "₹66,990",
            price = 66990.0,
            store = "Amazon India",
            deliverySpeed = "2 Days",
            badge = "Best Overall 🏆",
            imageUrl = "https://m.media-amazon.com/images/I/718yS2dhywL._SX679_.jpg",
            url = "https://www.amazon.in/s?k=Lenovo+LOQ+15",
            specs = mapOf("GPU" to "NVIDIA RTX 4050 6GB (95W TGP)", "Display" to "100% sRGB 144Hz", "RAM" to "16GB DDR5")
        )
        val tuf = OfflineComparisonItem(
            title = "ASUS TUF Gaming A15 (Ryzen 7, RTX 3050)",
            priceDisplay = "₹62,990",
            price = 62990.0,
            store = "Flipkart",
            deliverySpeed = "2 Days",
            badge = "Best Battery",
            imageUrl = "https://m.media-amazon.com/images/I/81FK83j+QCL._SX679_.jpg",
            url = "https://www.flipkart.com/search?q=ASUS+TUF+A15",
            specs = mapOf("Battery" to "90Wh (6-8 hours non-gaming)", "Chassis" to "MIL-STD-810H Military Tough")
        )
        return OfflineComparisonPayload(
            title = "Best Laptops Under 70k Comparison",
            headline = "💻 Laptops Under ₹70,000 Shootout",
            summaryMarkdown = """
                📊 **Laptops Under ₹70,000 Ranked:**
                - **Lenovo LOQ 15:** **₹66,990** (RTX 4050, 100% sRGB) 🏆 *Top Pick*
                - **ASUS TUF A15:** **₹62,990** (Ryzen 7, 90Wh Battery) *Best Battery Life*
            """.trimIndent(),
            recommendedItem = loq,
            items = listOf(loq, tuf),
            verdict = "Lenovo LOQ 15 offers a full RTX 4050 and 100% sRGB color-accurate display for ₹66,990."
        )
    }
}
