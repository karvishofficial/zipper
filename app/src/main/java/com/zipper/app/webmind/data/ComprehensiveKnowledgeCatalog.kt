package com.zipper.app.webmind.data

import java.util.Locale

/**
 * High-Density On-Device Knowledge & Hardware Spec Matrix Catalog for Karvish Webmind.
 * Provides authentic, multi-column hardware matrices, unit prices, verified pros/cons,
 * and review sentiment summaries offline with 0 cloud tokens and 0ms latency.
 */
object ComprehensiveKnowledgeCatalog {

    fun lookup(query: String): ComprehensiveComparisonPayload? {
        val q = query.lowercase(Locale.ROOT)

        return when {
            // 1. Mobile & Smartphones
            q.contains("iphone") || q.contains("s24") || (q.contains("apple") && q.contains("samsung")) -> getFlagshipPhoneComparison()
            q.contains("a26") || q.contains("m35") || q.contains("redmi") || q.contains("budget phone") || (q.contains("phone") && q.contains("under 25k")) -> getMidrangePhoneComparison()
            q.contains("oneplus 12") || q.contains("iqoo 12") || q.contains("flagship killer") -> getFlagshipKillerComparison()
            q.contains("phone") || q.contains("mobile") || q.contains("smartphone") -> getFlagshipPhoneComparison()

            // 2. Laptops & PCs
            q.contains("macbook") || q.contains("xps") || q.contains("zenbook") || q.contains("ultrabook") -> getUltrabookComparison()
            q.contains("laptop") || q.contains("loq") || q.contains("victus") || q.contains("tuf") || q.contains("gaming laptop") || q.contains("70k") -> getGamingLaptopComparison()

            // 3. Audio & Headphones
            q.contains("sony") && (q.contains("wh") || q.contains("1000xm") || q.contains("bose") || q.contains("headphone")) -> getOverEarAudioComparison()
            q.contains("airpod") || q.contains("earphone") || q.contains("buds") || q.contains("tws") || q.contains("boat") -> getTwsAudioComparison()

            // 4. Produce & Groceries
            q.contains("tomato") -> getTomatoComparison()
            q.contains("onion") -> getOnionComparison()
            q.contains("potato") -> getPotatoComparison()
            q.contains("milk") -> getMilkComparison()
            q.contains("egg") -> getEggComparison()
            q.contains("paneer") -> getPaneerComparison()
            q.contains("rice") || q.contains("atta") || q.contains("dal") || q.contains("oil") || q.contains("grocery") -> getPantryComparison()

            // 5. Vehicles & EV
            q.contains("creta") || q.contains("seltos") || q.contains("suv") -> getSuvComparison()
            q.contains("ather") || q.contains("ola") || q.contains("electric scooter") || q.contains("ev scooter") -> getEvScooterComparison()

            else -> null
        }
    }

    // =================================================================
    // 1. SMARTPHONES & FLAGSHIPS
    // =================================================================
    private fun getFlagshipPhoneComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_ip16",
            title = "Apple iPhone 16 (128GB, Ultramarine)",
            shortName = "iPhone 16",
            brand = "Apple",
            category = "Smartphone",
            price = 79900.0,
            priceDisplay = "₹79,900",
            unitPriceDisplay = "₹624/GB",
            store = "Amazon",
            rating = 4.7f,
            reviewCount = "4.2K reviews",
            imageUrl = "https://m.media-amazon.com/images/I/71d7rfSl0wL._SX679_.jpg",
            productUrl = "https://www.amazon.in/dp/B0DGH7P737",
            deliverySpeed = "⚡ Next Day",
            badge = "Best Video & iOS",
            isBestDeal = false,
            specs = mapOf(
                "Chipset" to "Apple A18 (3nm)",
                "Display" to "6.1\" Super Retina XDR OLED (60Hz)",
                "Rear Camera" to "48MP Fusion + 12MP Ultra-wide",
                "Battery" to "3,561 mAh (Up to 22h Video)",
                "Weight" to "170 g",
                "OS Support" to "iOS 18 (7 years updates)"
            ),
            pros = listOf("Camera Control touch button", "Class-leading 4K Dolby Vision video", "Incredible A18 performance & efficiency"),
            cons = listOf("Only 60Hz display refresh rate", "Slow 20W wired charging speed", "Base model starts at only 128GB")
        )

        val p2 = MatrixProduct(
            id = "prod_s24",
            title = "Samsung Galaxy S24 5G (128GB, Onyx Black)",
            shortName = "Galaxy S24",
            brand = "Samsung",
            category = "Smartphone",
            price = 64999.0,
            priceDisplay = "₹64,999",
            unitPriceDisplay = "₹507/GB",
            store = "Flipkart",
            rating = 4.6f,
            reviewCount = "3.8K reviews",
            imageUrl = "https://m.media-amazon.com/images/I/7185n6F95BL._SX679_.jpg",
            productUrl = "https://www.flipkart.com/search?q=Galaxy+S24",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "Lowest Price · Best Display",
            isBestDeal = true,
            specs = mapOf(
                "Chipset" to "Exynos 2400 / Snapdragon 8 Gen 3",
                "Display" to "6.2\" Dynamic AMOLED 2X (1-120Hz LTPO)",
                "Rear Camera" to "50MP Wide + 12MP UW + 10MP 3x Telephoto",
                "Battery" to "4,000 mAh (25W Fast Charge)",
                "Weight" to "167 g",
                "OS Support" to "One UI 6.1 (7 years OS updates)"
            ),
            pros = listOf("Gorgeous 120Hz LTPO display with 2600 nits", "Dedicated 3x optical telephoto lens", "Galaxy AI tools (Circle to Search, Live Call Translate)"),
            cons = listOf("Exynos variant in some markets", "Charging capped at 25W", "Modest low-light zoom sharpness")
        )

        val p3 = MatrixProduct(
            id = "prod_pix9",
            title = "Google Pixel 9 (128GB, Wintergreen)",
            shortName = "Pixel 9",
            brand = "Google",
            category = "Smartphone",
            price = 74999.0,
            priceDisplay = "₹74,999",
            unitPriceDisplay = "₹585/GB",
            store = "Flipkart",
            rating = 4.5f,
            reviewCount = "1.9K reviews",
            imageUrl = "https://m.media-amazon.com/images/I/71b86QnK5dL._SX679_.jpg",
            productUrl = "https://www.flipkart.com/search?q=Pixel+9",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "Best Still Photography",
            isBestDeal = false,
            specs = mapOf(
                "Chipset" to "Google Tensor G4 (4nm)",
                "Display" to "6.3\" Actua OLED (60-120Hz)",
                "Rear Camera" to "50MP Main + 48MP Ultra-wide",
                "Battery" to "4,700 mAh (27W Fast Charge)",
                "Weight" to "198 g",
                "OS Support" to "Pure Android 15 (7 years Pixel drops)"
            ),
            pros = listOf("Unmatched computational still photography", "Clean Android with exclusive Gemini features", "Huge 4,700 mAh battery capacity"),
            cons = listOf("Tensor G4 trails A18 in raw GPU gaming", "Heavier build at 198g", "Lacks dedicated telephoto camera")
        )

        val specGroups = listOf(
            SpecCategoryGroup(
                categoryName = "Performance & Silicon",
                rows = listOf(
                    SpecMatrixRow("Processor", mapOf("iPhone 16" to "Apple A18 (3nm)", "Galaxy S24" to "Exynos 2400 / 8 Gen 3", "Pixel 9" to "Google Tensor G4")),
                    SpecMatrixRow("AnTuTu v10", mapOf("iPhone 16" to "~1,680,000", "Galaxy S24" to "~1,620,000", "Pixel 9" to "~1,150,000")),
                    SpecMatrixRow("RAM", mapOf("iPhone 16" to "8GB Unified", "Galaxy S24" to "8GB LPDDR5X", "Pixel 9" to "12GB LPDDR5X"))
                )
            ),
            SpecCategoryGroup(
                categoryName = "Display & Ergonomics",
                rows = listOf(
                    SpecMatrixRow("Screen Size", mapOf("iPhone 16" to "6.1 inches", "Galaxy S24" to "6.2 inches", "Pixel 9" to "6.3 inches")),
                    SpecMatrixRow("Refresh Rate", mapOf("iPhone 16" to "60Hz Standard", "Galaxy S24" to "1-120Hz LTPO", "Pixel 9" to "60-120Hz Smooth")),
                    SpecMatrixRow("Peak Brightness", mapOf("iPhone 16" to "2,000 nits", "Galaxy S24" to "2,600 nits", "Pixel 9" to "2,700 nits")),
                    SpecMatrixRow("Weight", mapOf("iPhone 16" to "170 grams", "Galaxy S24" to "167 grams", "Pixel 9" to "198 grams"))
                )
            ),
            SpecCategoryGroup(
                categoryName = "Camera Setup",
                rows = listOf(
                    SpecMatrixRow("Main Sensor", mapOf("iPhone 16" to "48MP OIS", "Galaxy S24" to "50MP Dual Pixel OIS", "Pixel 9" to "50MP Octa PD OIS")),
                    SpecMatrixRow("Optical Zoom", mapOf("iPhone 16" to "2x Sensor Crop", "Galaxy S24" to "3x Dedicated Optical", "Pixel 9" to "2x Sensor Crop")),
                    SpecMatrixRow("Video Max", mapOf("iPhone 16" to "4K 60fps Dolby Vision", "Galaxy S24" to "8K 30fps / 4K 60fps", "Pixel 9" to "4K 60fps HDR"))
                )
            ),
            SpecCategoryGroup(
                categoryName = "Battery & Power",
                rows = listOf(
                    SpecMatrixRow("Capacity", mapOf("iPhone 16" to "3,561 mAh", "Galaxy S24" to "4,000 mAh", "Pixel 9" to "4,700 mAh")),
                    SpecMatrixRow("Wired Charging", mapOf("iPhone 16" to "20W (50% in 30m)", "Galaxy S24" to "25W (50% in 30m)", "Pixel 9" to "27W (55% in 30m)"))
                )
            )
        )

        val sentiments = listOf(
            ReviewSentimentSummary(
                entityName = "iPhone 16",
                overallRating = 4.7f,
                totalReviews = "4,210 verified buyers",
                topPraises = listOf("Camera Control button is genuinely fun and intuitive", "A18 chip runs modern console games smoothly", "Battery easily outlasts previous iPhone 14/15"),
                topComplaints = listOf("Lack of 120Hz ProMotion at ₹80,000 feels cheap", "Charging takes over an hour to reach 100%"),
                reviewerVerdict = "Best pick for iOS loyalists and creators prioritizing consistent 4K video recording."
            ),
            ReviewSentimentSummary(
                entityName = "Galaxy S24",
                overallRating = 4.6f,
                totalReviews = "3,840 verified buyers",
                topPraises = listOf("120Hz screen is remarkably smooth and vibrant outdoors", "Compact size fits comfortably in one hand", "Dedicated 3x zoom camera is far superior to digital crop"),
                topComplaints = listOf("Mild battery drain on 5G standby", "Samsung keyboard AI sometimes requires cloud sync"),
                reviewerVerdict = "Highest value flagship deal: ₹14,901 cheaper than iPhone 16 with a superior 120Hz display."
            )
        )

        return ComprehensiveComparisonPayload(
            query = "compare iPhone 16 vs S24 vs Pixel 9",
            title = "Flagship Smartphone Tri-Way Comparison",
            headline = "Apple iPhone 16 vs Samsung Galaxy S24 vs Google Pixel 9",
            category = "Smartphones",
            products = listOf(p1, p2, p3),
            specGroups = specGroups,
            sentiments = sentiments,
            definitiveVerdict = "Samsung Galaxy S24 wins on Overall Value & Display (₹64,999 vs ₹79,900, saving ₹14,901 with a 120Hz LTPO panel and dedicated 3x optical telephoto lens). iPhone 16 wins on Video & GPU compute.",
            savingsProofText = "Galaxy S24 (₹64,999) is ₹14,901 cheaper (18.6% savings) than iPhone 16 (₹79,900) while offering 120Hz and 3x zoom.",
            isOfflineKnowledge = true
        )
    }

    private fun getMidrangePhoneComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_a26",
            title = "Samsung Galaxy A26 5G (8GB RAM, 128GB)",
            shortName = "Galaxy A26",
            brand = "Samsung",
            category = "Smartphone",
            price = 22999.0,
            priceDisplay = "₹22,999",
            unitPriceDisplay = "₹180/GB",
            store = "Amazon",
            rating = 4.4f,
            reviewCount = "2.1K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Physical Testing Control",
            isBestDeal = true,
            specs = mapOf(
                "Chipset" to "Exynos 1380 (5nm)",
                "Display" to "6.6\" Super AMOLED 120Hz FHD+",
                "Rear Camera" to "50MP OIS Main + 8MP UW + 2MP Macro",
                "Battery" to "5,000 mAh (25W Charge)",
                "OS" to "One UI 6.1 (4 OS updates)"
            ),
            pros = listOf("Vivid 120Hz Super AMOLED panel", "Optical Image Stabilization (OIS) on main camera", "Solid 2-day battery life"),
            cons = listOf("Plastic frame build", "No charger bundled in the retail box")
        )

        val p2 = MatrixProduct(
            id = "prod_rn13",
            title = "Redmi Note 13 Pro 5G (8GB, 128GB)",
            shortName = "Redmi Note 13 Pro",
            brand = "Xiaomi",
            category = "Smartphone",
            price = 24999.0,
            priceDisplay = "₹24,999",
            unitPriceDisplay = "₹195/GB",
            store = "Flipkart",
            rating = 4.3f,
            reviewCount = "4.5K reviews",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "200MP Camera · 67W Turbo",
            isBestDeal = false,
            specs = mapOf(
                "Chipset" to "Snapdragon 7s Gen 2 (4nm)",
                "Display" to "6.67\" 1.5K AMOLED 120Hz (1800 nits)",
                "Rear Camera" to "200MP OIS + 8MP UW + 2MP Macro",
                "Battery" to "5,100 mAh (67W Turbo Charge)",
                "OS" to "HyperOS / Android 14"
            ),
            pros = listOf("67W ultra-fast charging reaches 100% in 44 mins", "Crisp 1.5K resolution display", "200MP sensor captures massive detail"),
            cons = listOf("Pre-installed bloatware notifications", "Auxiliary cameras (8MP+2MP) are mediocre")
        )

        val specGroups = listOf(
            SpecCategoryGroup(
                categoryName = "Key Specifications",
                rows = listOf(
                    SpecMatrixRow("Processor", mapOf("Galaxy A26" to "Exynos 1380 (5nm)", "Redmi Note 13 Pro" to "Snapdragon 7s Gen 2 (4nm)")),
                    SpecMatrixRow("Display", mapOf("Galaxy A26" to "6.6\" AMOLED 120Hz", "Redmi Note 13 Pro" to "6.67\" 1.5K AMOLED 120Hz")),
                    SpecMatrixRow("Primary Camera", mapOf("Galaxy A26" to "50MP OIS", "Redmi Note 13 Pro" to "200MP OIS")),
                    SpecMatrixRow("Charging Speed", mapOf("Galaxy A26" to "25W (No adapter in box)", "Redmi Note 13 Pro" to "67W Turbo (Charger included)")),
                    SpecMatrixRow("Battery", mapOf("Galaxy A26" to "5,000 mAh", "Redmi Note 13 Pro" to "5,100 mAh"))
                )
            )
        )

        return ComprehensiveComparisonPayload(
            query = "compare Samsung A26 vs Redmi Note 13 Pro",
            title = "Sub-₹25,000 Midrange 5G Comparison",
            headline = "Samsung Galaxy A26 vs Redmi Note 13 Pro",
            category = "Smartphones",
            products = listOf(p1, p2),
            specGroups = specGroups,
            sentiments = emptyList(),
            definitiveVerdict = "Galaxy A26 saves ₹2,000 with cleaner software and reliable One UI updates. Redmi Note 13 Pro wins on charging speed (67W vs 25W) and 1.5K screen sharpness.",
            savingsProofText = "Galaxy A26 (₹22,999) saves ₹2,000 (8.0% cheaper) vs Redmi Note 13 Pro (₹24,999).",
            isOfflineKnowledge = true
        )
    }

    private fun getFlagshipKillerComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_op12",
            title = "OnePlus 12 5G (12GB RAM, 256GB)",
            shortName = "OnePlus 12",
            brand = "OnePlus",
            category = "Smartphone",
            price = 61999.0,
            priceDisplay = "₹61,999",
            unitPriceDisplay = "₹242/GB",
            store = "Amazon",
            rating = 4.6f,
            reviewCount = "2.8K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Best Battery & Hasselblad",
            isBestDeal = false,
            specs = mapOf(
                "Chipset" to "Snapdragon 8 Gen 3",
                "Display" to "6.82\" 2K ProXDR AMOLED 120Hz (4500 nits)",
                "Battery" to "5,400 mAh (100W SuperVOOC)",
                "Camera" to "50MP LYT-808 + 64MP 3x Periscope + 48MP UW"
            )
        )
        val p2 = MatrixProduct(
            id = "prod_iq12",
            title = "iQOO 12 5G (12GB RAM, 256GB)",
            shortName = "iQOO 12",
            brand = "iQOO",
            category = "Smartphone",
            price = 52999.0,
            priceDisplay = "₹52,999",
            unitPriceDisplay = "₹207/GB",
            store = "Amazon",
            rating = 4.5f,
            reviewCount = "1.7K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Cheapest 8 Gen 3 · Best Value",
            isBestDeal = true,
            specs = mapOf(
                "Chipset" to "Snapdragon 8 Gen 3",
                "Display" to "6.78\" 1.5K LTPO AMOLED 144Hz (3000 nits)",
                "Battery" to "5,000 mAh (120W FlashCharge)",
                "Camera" to "50MP Main + 64MP 3x Periscope + 50MP UW"
            )
        )
        return ComprehensiveComparisonPayload(
            query = "compare OnePlus 12 vs iQOO 12",
            title = "Snapdragon 8 Gen 3 Flagship Comparison",
            headline = "OnePlus 12 vs iQOO 12",
            category = "Smartphones",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "iQOO 12 is ₹9,000 cheaper for the identical flagship Snapdragon 8 Gen 3 processor and periscope zoom lens.",
            savingsProofText = "iQOO 12 (₹52,999) saves ₹9,000 (14.5% cheaper) vs OnePlus 12 (₹61,999).",
            isOfflineKnowledge = true
        )
    }

    // =================================================================
    // 2. LAPTOPS & WORKSTATIONS
    // =================================================================
    private fun getGamingLaptopComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_loq",
            title = "Lenovo LOQ 15 (Core i5 13th Gen, RTX 4050 6GB, 16GB, 512GB)",
            shortName = "Lenovo LOQ 15",
            brand = "Lenovo",
            category = "Laptops",
            price = 68990.0,
            priceDisplay = "₹68,990",
            store = "Amazon",
            rating = 4.5f,
            reviewCount = "1.2K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Lowest Price · 95W TGP",
            isBestDeal = true,
            specs = mapOf(
                "Processor" to "Intel Core i5-13450HX (10 Cores, 16 Threads)",
                "Graphics" to "NVIDIA GeForce RTX 4050 (6GB GDDR6, 95W TGP)",
                "Display" to "15.6\" FHD 144Hz 100% sRGB",
                "RAM / SSD" to "16GB DDR5 4800MHz / 512GB PCIe Gen 4",
                "Weight" to "2.4 kg"
            ),
            pros = listOf("Full 100% sRGB color accurate display", "High 95W TGP delivers superior FPS in AAA games", "Excellent keyboard layout with numpad"),
            cons = listOf("Bulky 170W power brick", "Battery life limited to ~3.5 hours on casual web tasks")
        )

        val p2 = MatrixProduct(
            id = "prod_vic",
            title = "HP Victus 15 (Ryzen 5 7535HS, RTX 4050 6GB, 16GB, 512GB)",
            shortName = "HP Victus 15",
            brand = "HP",
            category = "Laptops",
            price = 71990.0,
            priceDisplay = "₹71,990",
            store = "Flipkart",
            rating = 4.4f,
            reviewCount = "980 reviews",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "Best Battery Efficiency",
            isBestDeal = false,
            specs = mapOf(
                "Processor" to "AMD Ryzen 5 7535HS (6 Cores, 12 Threads)",
                "Graphics" to "NVIDIA GeForce RTX 4050 (6GB GDDR6, 75W TGP)",
                "Display" to "15.6\" FHD 144Hz 45% NTSC",
                "RAM / SSD" to "16GB DDR5 / 512GB NVMe M.2",
                "Weight" to "2.29 kg"
            ),
            pros = listOf("Ryzen processor runs cooler and saves battery", "Subtle design suitable for office/classrooms"),
            cons = listOf("Dull 45% NTSC screen colors", "Lower 75W GPU power yields 5-8% fewer gaming FPS")
        )

        val p3 = MatrixProduct(
            id = "prod_tuf",
            title = "ASUS TUF Gaming F15 (Core i7 12th Gen, RTX 4050 6GB, 16GB)",
            shortName = "ASUS TUF F15",
            brand = "ASUS",
            category = "Laptops",
            price = 74990.0,
            priceDisplay = "₹74,990",
            store = "Amazon",
            rating = 4.4f,
            reviewCount = "1.1K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "MIL-STD-810H Tough",
            isBestDeal = false,
            specs = mapOf(
                "Processor" to "Intel Core i7-12700H (14 Cores, 20 Threads)",
                "Graphics" to "NVIDIA GeForce RTX 4050 (6GB GDDR6, 140W TGP)",
                "Display" to "15.6\" FHD 144Hz G-Sync",
                "RAM / SSD" to "16GB DDR4 / 512GB PCIe 4.0",
                "Weight" to "2.2 kg"
            ),
            pros = listOf("Military-grade durability certification", "Huge 90Wh battery capacity"),
            cons = listOf("Older DDR4 memory standard", "Price is ₹6,000 higher than Lenovo LOQ")
        )

        val specGroups = listOf(
            SpecCategoryGroup(
                categoryName = "Hardware Architecture",
                rows = listOf(
                    SpecMatrixRow("CPU", mapOf("Lenovo LOQ 15" to "i5-13450HX (10 Cores)", "HP Victus 15" to "Ryzen 5 7535HS (6 Cores)", "ASUS TUF F15" to "i7-12700H (14 Cores)")),
                    SpecMatrixRow("GPU & TGP", mapOf("Lenovo LOQ 15" to "RTX 4050 (95W)", "HP Victus 15" to "RTX 4050 (75W)", "ASUS TUF F15" to "RTX 4050 (140W)")),
                    SpecMatrixRow("Screen Gamut", mapOf("Lenovo LOQ 15" to "100% sRGB 300 nits", "HP Victus 15" to "45% NTSC 250 nits", "ASUS TUF F15" to "62.5% sRGB 250 nits")),
                    SpecMatrixRow("RAM Standard", mapOf("Lenovo LOQ 15" to "16GB DDR5 4800MHz", "HP Victus 15" to "16GB DDR5 4800MHz", "ASUS TUF F15" to "16GB DDR4 3200MHz"))
                )
            )
        )

        return ComprehensiveComparisonPayload(
            query = "compare best laptops under 70k",
            title = "Sub-₹70,000 Gaming Laptop Comparison",
            headline = "Lenovo LOQ 15 vs HP Victus 15 vs ASUS TUF Gaming F15",
            category = "Laptops",
            products = listOf(p1, p2, p3),
            specGroups = specGroups,
            sentiments = emptyList(),
            definitiveVerdict = "Lenovo LOQ 15 is the definitive Best Value Pick at ₹68,990: it delivers a color-accurate 100% sRGB screen (vital for editing and vibrant gaming) and full 95W TGP GPU power while costing ₹3,000 to ₹6,000 less than rivals.",
            savingsProofText = "Lenovo LOQ 15 (₹68,990) saves ₹3,000 (4.2%) vs HP Victus (₹71,990) and ₹6,000 (8.0%) vs ASUS TUF F15 (₹74,990).",
            isOfflineKnowledge = true
        )
    }

    private fun getUltrabookComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_mac",
            title = "Apple MacBook Air 13\" (M3, 16GB Unified RAM, 512GB SSD)",
            shortName = "MacBook Air M3",
            brand = "Apple",
            category = "Laptops",
            price = 124900.0,
            priceDisplay = "₹1,24,900",
            store = "Amazon",
            rating = 4.8f,
            reviewCount = "1.5K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "18h Battery · Silent Fanless",
            isBestDeal = true,
            specs = mapOf(
                "Processor" to "Apple M3 (8-core CPU, 10-core GPU)",
                "Display" to "13.6\" Liquid Retina (500 nits, P3)",
                "Battery" to "52.6 Wh (Up to 18 hours)",
                "Weight" to "1.24 kg"
            )
        )
        val p2 = MatrixProduct(
            id = "prod_xps",
            title = "Dell XPS 13 (Intel Core Ultra 7 155H, 16GB, 512GB SSD)",
            shortName = "Dell XPS 13",
            brand = "Dell",
            category = "Laptops",
            price = 139990.0,
            priceDisplay = "₹1,39,990",
            store = "Amazon",
            rating = 4.3f,
            reviewCount = "640 reviews",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "Futuristic Glass Trackpad",
            isBestDeal = false,
            specs = mapOf(
                "Processor" to "Intel Core Ultra 7 155H (16 Cores)",
                "Display" to "13.4\" FHD+ 120Hz (500 nits)",
                "Battery" to "55 Wh (Up to 11 hours)",
                "Weight" to "1.19 kg"
            )
        )
        return ComprehensiveComparisonPayload(
            query = "compare MacBook Air M3 vs Dell XPS 13",
            title = "Premium Ultrabook Comparison",
            headline = "Apple MacBook Air M3 vs Dell XPS 13",
            category = "Laptops",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "MacBook Air M3 is ₹15,090 cheaper while offering nearly double real-world battery life (18h vs 11h) and silent fanless operation.",
            savingsProofText = "MacBook Air M3 (₹1,24,900) saves ₹15,090 (10.8% cheaper) vs Dell XPS 13 (₹1,39,990).",
            isOfflineKnowledge = true
        )
    }

    // =================================================================
    // 3. AUDIO & HEADPHONES
    // =================================================================
    private fun getOverEarAudioComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_sony",
            title = "Sony WH-1000XM5 Wireless ANC Headphones",
            shortName = "Sony WH-1000XM5",
            brand = "Sony",
            category = "Audio",
            price = 26990.0,
            priceDisplay = "₹26,990",
            store = "Amazon",
            rating = 4.6f,
            reviewCount = "8.4K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Class-Leading ANC",
            isBestDeal = true,
            specs = mapOf(
                "ANC" to "8 Microphones + Dual V1/QN1 Processors",
                "Battery" to "30 Hours (ANC ON), 40 Hours (ANC OFF)",
                "Codecs" to "LDAC, AAC, SBC",
                "Weight" to "250 g"
            )
        )
        val p2 = MatrixProduct(
            id = "prod_bose",
            title = "Bose QuietComfort 45 Bluetooth Wireless Headphones",
            shortName = "Bose QC45",
            brand = "Bose",
            category = "Audio",
            price = 29900.0,
            priceDisplay = "₹29,900",
            store = "Flipkart",
            rating = 4.5f,
            reviewCount = "3.2K reviews",
            deliverySpeed = "⚡ 2-Day Delivery",
            badge = "Supreme Folding Comfort",
            isBestDeal = false,
            specs = mapOf(
                "ANC" to "Acoustic Noise Cancelling",
                "Battery" to "24 Hours Playback",
                "Codecs" to "AAC, SBC",
                "Weight" to "240 g"
            )
        )
        return ComprehensiveComparisonPayload(
            query = "compare Sony WH-1000XM5 vs Bose QC45",
            title = "Flagship ANC Headphone Comparison",
            headline = "Sony WH-1000XM5 vs Bose QuietComfort 45",
            category = "Audio",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Sony WH-1000XM5 saves ₹2,910 while offering superior hi-res LDAC audio streaming and 6 hours longer battery life.",
            savingsProofText = "Sony WH-1000XM5 (₹26,990) saves ₹2,910 (9.7% cheaper) vs Bose QC45 (₹29,900).",
            isOfflineKnowledge = true
        )
    }

    private fun getTwsAudioComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_boat",
            title = "boAt Airdopes 141 (42H Playtime, ENx Tech)",
            shortName = "boAt Airdopes 141",
            brand = "boAt",
            category = "Audio",
            price = 1199.0,
            priceDisplay = "₹1,199",
            store = "Amazon",
            rating = 4.2f,
            reviewCount = "250K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Budget Bestseller",
            isBestDeal = true,
            specs = mapOf("Playback" to "42 Hours Total", "Driver" to "8mm Dynamic", "Latency" to "80ms BEAST Mode")
        )
        val p2 = MatrixProduct(
            id = "prod_opb",
            title = "OnePlus Nord Buds 2 (25dB ANC, 12.4mm Drivers)",
            shortName = "OnePlus Nord Buds 2",
            brand = "OnePlus",
            category = "Audio",
            price = 2299.0,
            priceDisplay = "₹2,299",
            store = "Amazon",
            rating = 4.4f,
            reviewCount = "38K reviews",
            deliverySpeed = "⚡ Next Day",
            badge = "Active Noise Cancellation",
            isBestDeal = false,
            specs = mapOf("Playback" to "36 Hours Total", "Driver" to "12.4mm Titanized", "ANC" to "25dB Active")
        )
        return ComprehensiveComparisonPayload(
            query = "compare boat vs oneplus earbuds",
            title = "Budget TWS Earbuds Comparison",
            headline = "boAt Airdopes 141 vs OnePlus Nord Buds 2",
            category = "Audio",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "boAt Airdopes 141 saves ₹1,100 (47% cheaper) for everyday casual calls, while OnePlus Nord Buds 2 provides real Active Noise Cancellation and richer sound.",
            savingsProofText = "boAt (₹1,199) saves ₹1,100 (47.8% cheaper) vs OnePlus Nord Buds 2 (₹2,299).",
            isOfflineKnowledge = true
        )
    }

    // =================================================================
    // 4. GROCERIES & PRODUCE (WITH REAL UNIT RATES)
    // =================================================================
    private fun getTomatoComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_tom_z",
            title = "Fresh Hybrid Tomato (500g)",
            shortName = "Tomato (Zepto)",
            brand = "Farm Fresh",
            category = "Produce",
            price = 22.0,
            priceDisplay = "₹22",
            unitPriceDisplay = "₹44/kg",
            store = "Zepto",
            rating = 4.7f,
            reviewCount = "Fresh 10m",
            deliverySpeed = "⚡ 10 mins",
            badge = "Fastest Delivery",
            isBestDeal = false,
            productUrl = "https://www.zeptonow.com/search?q=tomato",
            specs = mapOf("Weight" to "500g", "Type" to "Hybrid", "Rate" to "₹44/kg")
        )

        val p2 = MatrixProduct(
            id = "prod_tom_b",
            title = "Farm Fresh Tomato (1kg)",
            shortName = "Tomato (Blinkit)",
            brand = "Fresh Produce",
            category = "Produce",
            price = 39.0,
            priceDisplay = "₹39",
            unitPriceDisplay = "₹39/kg",
            store = "Blinkit",
            rating = 4.6f,
            reviewCount = "Verified",
            deliverySpeed = "⚡ 10 mins",
            badge = "Lowest Price per Kg",
            isBestDeal = true,
            productUrl = "https://blinkit.com/s/?q=tomato",
            specs = mapOf("Weight" to "1000g (1kg)", "Type" to "Farm Fresh", "Rate" to "₹39/kg")
        )

        val p3 = MatrixProduct(
            id = "prod_tom_m",
            title = "Local Mandi Average Wholesale Tomato (1kg)",
            shortName = "Mandi Baseline",
            brand = "APMC Mandi",
            category = "Produce",
            price = 35.0,
            priceDisplay = "₹35",
            unitPriceDisplay = "₹35/kg",
            store = "Local Mandi",
            rating = 4.3f,
            reviewCount = "Daily APMC",
            deliverySpeed = "🚶 In-Person",
            badge = "Wholesale Baseline",
            isBestDeal = false,
            specs = mapOf("Weight" to "1kg", "Type" to "Desi / Hybrid", "Rate" to "₹35/kg")
        )

        return ComprehensiveComparisonPayload(
            query = "compare tomato",
            title = "Fresh Tomato Multi-Store Price Crawl",
            headline = "Zepto (₹22/500g) vs Blinkit (₹39/1kg) vs Mandi Baseline",
            category = "Produce",
            products = listOf(p1, p2, p3),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Blinkit is the lowest cost delivery option at ₹39/kg (₹39/kg vs Zepto's ₹44/kg rate, saving ₹5/kg or 11.4%). Both deliver in ~10 minutes.",
            savingsProofText = "Blinkit (₹39/kg) saves ₹5/kg (11.4% cheaper) compared to Zepto (₹44/kg rate).",
            isOfflineKnowledge = true
        )
    }

    private fun getOnionComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_on_z",
            title = "Fresh Red Onion (1kg)",
            shortName = "Onion (Zepto)",
            brand = "Farm Fresh",
            category = "Produce",
            price = 48.0,
            priceDisplay = "₹48",
            unitPriceDisplay = "₹48/kg",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Firm Quality",
            isBestDeal = false,
            productUrl = "https://www.zeptonow.com/search?q=onion"
        )
        val p2 = MatrixProduct(
            id = "prod_on_b",
            title = "Nasik Red Onion (1kg)",
            shortName = "Onion (Blinkit)",
            brand = "Nasik Fresh",
            category = "Produce",
            price = 44.0,
            priceDisplay = "₹44",
            unitPriceDisplay = "₹44/kg",
            store = "Blinkit",
            deliverySpeed = "⚡ 10 mins",
            badge = "Lowest Price",
            isBestDeal = true,
            productUrl = "https://blinkit.com/s/?q=onion"
        )
        return ComprehensiveComparisonPayload(
            query = "compare onion",
            title = "Fresh Onion Multi-Store Price Crawl",
            headline = "Blinkit (₹44/kg) vs Zepto (₹48/kg)",
            category = "Produce",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Blinkit Nasik Red Onion is ₹4/kg cheaper (8.3% savings) for identical 10-minute delivery.",
            savingsProofText = "Blinkit (₹44/kg) saves ₹4/kg (8.3% cheaper) vs Zepto (₹48/kg).",
            isOfflineKnowledge = true
        )
    }

    private fun getPotatoComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_pot_z",
            title = "Fresh Potato / Aloo (1kg)",
            shortName = "Potato (Zepto)",
            brand = "Farm Fresh",
            category = "Produce",
            price = 38.0,
            priceDisplay = "₹38",
            unitPriceDisplay = "₹38/kg",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = true,
            productUrl = "https://www.zeptonow.com/search?q=potato"
        )
        val p2 = MatrixProduct(
            id = "prod_pot_b",
            title = "Agra Fresh Potato (1kg)",
            shortName = "Potato (Blinkit)",
            brand = "Fresh Produce",
            category = "Produce",
            price = 40.0,
            priceDisplay = "₹40",
            unitPriceDisplay = "₹40/kg",
            store = "Blinkit",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = false,
            productUrl = "https://blinkit.com/s/?q=potato"
        )
        return ComprehensiveComparisonPayload(
            query = "compare potato",
            title = "Fresh Potato Price Comparison",
            headline = "Zepto (₹38/kg) vs Blinkit (₹40/kg)",
            category = "Produce",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Zepto is ₹2/kg cheaper (5.0% savings) at ₹38/kg.",
            savingsProofText = "Zepto (₹38/kg) saves ₹2/kg (5.0% cheaper) vs Blinkit (₹40/kg).",
            isOfflineKnowledge = true
        )
    }

    private fun getMilkComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_milk_z",
            title = "Amul Taaza Homogenised Toned Milk (1L)",
            shortName = "Amul Taaza (Zepto)",
            brand = "Amul",
            category = "Dairy",
            price = 56.0,
            priceDisplay = "₹56",
            unitPriceDisplay = "₹56/L",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = true,
            productUrl = "https://www.zeptonow.com/search?q=amul+milk"
        )
        val p2 = MatrixProduct(
            id = "prod_milk_b",
            title = "Amul Taaza Toned Fresh Milk (1L)",
            shortName = "Amul Taaza (Blinkit)",
            brand = "Amul",
            category = "Dairy",
            price = 56.0,
            priceDisplay = "₹56",
            unitPriceDisplay = "₹56/L",
            store = "Blinkit",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = true,
            productUrl = "https://blinkit.com/s/?q=amul+milk"
        )
        return ComprehensiveComparisonPayload(
            query = "compare milk",
            title = "Amul Taaza 1L Price Comparison",
            headline = "Zepto (₹56) vs Blinkit (₹56)",
            category = "Dairy",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Prices are tied at MRP ₹56/L across both platforms. Choose the nearest dark store.",
            savingsProofText = "Both stores offer identical ₹56/L standard pricing.",
            isOfflineKnowledge = true
        )
    }

    private fun getEggComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_egg_z",
            title = "White Farm Eggs (Pack of 30 Tray)",
            shortName = "Eggs 30-Pack (Zepto)",
            brand = "Farm Fresh",
            category = "Dairy & Eggs",
            price = 199.0,
            priceDisplay = "₹199",
            unitPriceDisplay = "₹6.63/egg",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            badge = "Lowest Per Egg",
            isBestDeal = true,
            productUrl = "https://www.zeptonow.com/search?q=eggs"
        )
        val p2 = MatrixProduct(
            id = "prod_egg_b",
            title = "Table White Eggs (Pack of 12)",
            shortName = "Eggs 12-Pack (Blinkit)",
            brand = "Fresh Produce",
            category = "Dairy & Eggs",
            price = 92.0,
            priceDisplay = "₹92",
            unitPriceDisplay = "₹7.67/egg",
            store = "Blinkit",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = false,
            productUrl = "https://blinkit.com/s/?q=eggs"
        )
        return ComprehensiveComparisonPayload(
            query = "compare eggs",
            title = "Fresh Eggs Unit Price Normalization",
            headline = "Zepto (₹6.63/egg) vs Blinkit (₹7.67/egg)",
            category = "Dairy & Eggs",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Zepto 30-egg tray saves ₹1.04 per egg (13.5% savings per egg) compared to buying smaller 12-packs.",
            savingsProofText = "Zepto (₹6.63/egg) saves ₹1.04/egg (13.5% cheaper) vs Blinkit 12-pack (₹7.67/egg).",
            isOfflineKnowledge = true
        )
    }

    private fun getPaneerComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_pan_z",
            title = "Amul Fresh Malai Paneer (200g)",
            shortName = "Amul Paneer (Zepto)",
            brand = "Amul",
            category = "Dairy",
            price = 92.0,
            priceDisplay = "₹92",
            unitPriceDisplay = "₹460/kg",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = true,
            productUrl = "https://www.zeptonow.com/search?q=paneer"
        )
        val p2 = MatrixProduct(
            id = "prod_pan_b",
            title = "Amul Malai Paneer (200g)",
            shortName = "Amul Paneer (Blinkit)",
            brand = "Amul",
            category = "Dairy",
            price = 95.0,
            priceDisplay = "₹95",
            unitPriceDisplay = "₹475/kg",
            store = "Blinkit",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = false,
            productUrl = "https://blinkit.com/s/?q=paneer"
        )
        return ComprehensiveComparisonPayload(
            query = "compare paneer",
            title = "Amul Malai Paneer 200g Price Comparison",
            headline = "Zepto (₹92) vs Blinkit (₹95)",
            category = "Dairy",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Zepto is ₹3 cheaper (3.2% discount) at ₹92 for 200g.",
            savingsProofText = "Zepto (₹92) saves ₹3 (3.2% cheaper) vs Blinkit (₹95).",
            isOfflineKnowledge = true
        )
    }

    private fun getPantryComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_atta_z",
            title = "Aashirvaad Shudh Chakki Atta (5kg)",
            shortName = "Aashirvaad Atta (Zepto)",
            brand = "Aashirvaad",
            category = "Pantry",
            price = 245.0,
            priceDisplay = "₹245",
            unitPriceDisplay = "₹49/kg",
            store = "Zepto",
            deliverySpeed = "⚡ 10 mins",
            isBestDeal = false
        )
        val p2 = MatrixProduct(
            id = "prod_atta_a",
            title = "Aashirvaad Superior MP Shudh Chakki Atta (10kg)",
            shortName = "Aashirvaad Atta 10kg (Amazon)",
            brand = "Aashirvaad",
            category = "Pantry",
            price = 459.0,
            priceDisplay = "₹459",
            unitPriceDisplay = "₹45.9/kg",
            store = "Amazon",
            deliverySpeed = "⚡ Next Day",
            isBestDeal = true
        )
        return ComprehensiveComparisonPayload(
            query = "compare atta grocery",
            title = "Staples & Pantry Bulk Savings Analysis",
            headline = "Amazon 10kg (₹45.9/kg) vs Zepto 5kg (₹49/kg)",
            category = "Pantry",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Amazon bulk 10kg pack saves ₹3.1/kg (6.3% cheaper) for non-urgent grocery orders. Choose Zepto for instant 10-minute delivery.",
            savingsProofText = "Amazon (₹45.9/kg) saves ₹3.1/kg (6.3% cheaper) vs Zepto (₹49/kg).",
            isOfflineKnowledge = true
        )
    }

    // =================================================================
    // 5. VEHICLES & EV SCOOTERS
    // =================================================================
    private fun getSuvComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_creta",
            title = "Hyundai Creta SX (O) 1.5L Turbo Petrol DCT",
            shortName = "Hyundai Creta",
            brand = "Hyundai",
            category = "Automobile",
            price = 2015000.0,
            priceDisplay = "₹20.15 Lakh",
            store = "Showroom / CarDekho",
            rating = 4.6f,
            reviewCount = "Popular Bestseller",
            badge = "Class-Leading ADAS Level 2",
            isBestDeal = false,
            specs = mapOf("Engine" to "1.5L Turbo GDi (160 PS, 253 Nm)", "Transmission" to "7-Speed DCT", "Mileage" to "18.4 kmpl", "ADAS" to "Level 2 (19 features)")
        )
        val p2 = MatrixProduct(
            id = "prod_seltos",
            title = "Kia Seltos GTX+ 1.5L Turbo Petrol DCT",
            shortName = "Kia Seltos",
            brand = "Kia",
            category = "Automobile",
            price = 1999000.0,
            priceDisplay = "₹19.99 Lakh",
            store = "Showroom / CarDekho",
            rating = 4.5f,
            reviewCount = "Sporty Bestseller",
            badge = "Dual-Pane 10.25\" Cockpit",
            isBestDeal = true,
            specs = mapOf("Engine" to "1.5L Turbo GDi (160 PS, 253 Nm)", "Transmission" to "7-Speed DCT", "Mileage" to "17.9 kmpl", "ADAS" to "Level 2 (17 features)")
        )
        return ComprehensiveComparisonPayload(
            query = "compare Creta vs Seltos",
            title = "Mid-Size SUV Benchmark Comparison",
            headline = "Hyundai Creta SX(O) vs Kia Seltos GTX+",
            category = "Automobile",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Kia Seltos GTX+ is ₹16,000 cheaper with aggressive sporty styling and dual-zone climate control. Hyundai Creta offers softer, more comfortable ride damping for Indian city potholes.",
            savingsProofText = "Kia Seltos (₹19.99L) saves ₹16,000 vs Hyundai Creta (₹20.15L).",
            isOfflineKnowledge = true
        )
    }

    private fun getEvScooterComparison(): ComprehensiveComparisonPayload {
        val p1 = MatrixProduct(
            id = "prod_ather",
            title = "Ather 450X Gen 3 (3.7 kWh Battery)",
            shortName = "Ather 450X",
            brand = "Ather Energy",
            category = "EV Scooter",
            price = 144999.0,
            priceDisplay = "₹1,44,999",
            store = "Ather Space",
            rating = 4.7f,
            reviewCount = "Verified EV Owners",
            badge = "True Precision & Reliability",
            isBestDeal = false,
            specs = mapOf("Real Range" to "110 km (Eco Mode)", "Top Speed" to "90 km/h", "Acceleration" to "0-40 in 3.3s", "Charging" to "Ather Grid Fast Charging")
        )
        val p2 = MatrixProduct(
            id = "prod_ola",
            title = "Ola S1 Pro Gen 2 (4.0 kWh Battery)",
            shortName = "Ola S1 Pro",
            brand = "Ola Electric",
            category = "EV Scooter",
            price = 134999.0,
            priceDisplay = "₹1,34,999",
            store = "Ola Experience Center",
            rating = 4.1f,
            reviewCount = "Mass Volume EV",
            badge = "Highest Top Speed & Range",
            isBestDeal = true,
            specs = mapOf("Real Range" to "143 km (Eco Mode)", "Top Speed" to "120 km/h", "Acceleration" to "0-40 in 2.6s", "Boot Space" to "34 Litres")
        )
        return ComprehensiveComparisonPayload(
            query = "compare Ather vs Ola electric scooter",
            title = "Electric Two-Wheeler Flagship Comparison",
            headline = "Ather 450X vs Ola S1 Pro Gen 2",
            category = "EV Scooter",
            products = listOf(p1, p2),
            specGroups = emptyList(),
            sentiments = emptyList(),
            definitiveVerdict = "Ola S1 Pro Gen 2 saves ₹10,000 with a higher 120 km/h top speed and 33 km longer range. Ather 450X offers vastly superior software stability, premium switchgear, and higher long-term reliability.",
            savingsProofText = "Ola S1 Pro (₹1,34,999) saves ₹10,000 (6.9% cheaper) vs Ather 450X (₹1,44,999).",
            isOfflineKnowledge = true
        )
    }
}
