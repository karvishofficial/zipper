package com.zipper.app

import com.zipper.app.core.intelligence.scheduler.SmartScheduleValidator
import com.zipper.app.core.intelligence.store.CommitmentCategory
import com.zipper.app.core.intelligence.store.CommitmentEntity
import com.zipper.app.core.intelligence.store.CommitmentStatus
import com.zipper.app.core.intelligence.store.ContextMessage
import com.zipper.app.core.intelligence.store.LifeRhythmPrefs
import com.zipper.app.core.voice.CallReminder
import com.zipper.app.core.voice.CallReminderActionType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar

class PersonalIntelligenceSystemTests {

    @Test
    fun testCallReminderWithCommitmentMetadataJsonRoundTrip() {
        val original = CallReminder(
            id = "life_plan_101",
            title = "Evening Workout & Mobility",
            spokenPrompt = "Hey Karthik, your workout session starts in 5 minutes! Which workout app would you like to open?",
            triggerTimeEpochMs = 1790000000000L,
            actionType = CallReminderActionType.VOICE_ONLY,
            targetPackage = "fit.cure.android",
            commitmentId = 101L,
            category = "FITNESS",
            candidatePackages = listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"),
            sourcePerson = "Karvish"
        )

        val json = original.toJson()
        val restored = CallReminder.fromJson(json)

        assertEquals(original.id, restored.id)
        assertEquals(original.title, restored.title)
        assertEquals(original.spokenPrompt, restored.spokenPrompt)
        assertEquals(original.commitmentId, restored.commitmentId)
        assertEquals("FITNESS", restored.category)
        assertEquals(3, restored.candidatePackages.size)
        assertEquals("fit.cure.android", restored.candidatePackages[0])
        assertEquals("com.sec.android.app.shealth", restored.candidatePackages[1])
        assertEquals("com.strava", restored.candidatePackages[2])
        assertEquals("Karvish", restored.sourcePerson)
    }

    @Test
    fun testCollegeHoursProtection() {
        // College hours: 9 AM (9:00) to 6 PM (18:00)
        // Transit buffer: 30 minutes (free evening starts at 18:30)
        val prefs = LifeRhythmPrefs().apply {
            collegeStartHour = 9
            collegeEndHour = 18
            transitBufferMinutes = 30
        }
        val validator = SmartScheduleValidator(prefs)

        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        // Task proposed right in the middle of college hours: 11:30 AM
        val middayTask = CommitmentEntity(
            id = 1L,
            title = "Gym Session",
            category = CommitmentCategory.FITNESS,
            scheduledTimeMs = dayStartMs + (11 * 3600000L) + (30 * 60000L),
            durationMinutes = 45
        )

        val results = validator.validateAndRebalance(listOf(middayTask), dayStartMs)
        assertEquals(1, results.size)
        val validated = results[0]

        // Must be shifted past transit buffer (18:30 = 6:30 PM)
        val expectedFreeEveningMs = dayStartMs + (18 * 3600000L) + (30 * 60000L)
        assertTrue(
            "Task during college must be shifted to at least 18:30, was ${formatTime(validated.scheduledTimeMs)}",
            validated.scheduledTimeMs >= expectedFreeEveningMs
        )
    }

    @Test
    fun testTransitBufferProtection() {
        val prefs = LifeRhythmPrefs().apply {
            collegeStartHour = 9
            collegeEndHour = 18
            transitBufferMinutes = 30
        }
        val validator = SmartScheduleValidator(prefs)

        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        // Task proposed at 6:10 PM (inside 6:00 - 6:30 transit buffer)
        val commuteTask = CommitmentEntity(
            id = 2L,
            title = "Call with Team",
            category = CommitmentCategory.COMMUNICATION,
            scheduledTimeMs = dayStartMs + (18 * 3600000L) + (10 * 60000L),
            durationMinutes = 20
        )

        val results = validator.validateAndRebalance(listOf(commuteTask), dayStartMs)
        assertEquals(1, results.size)

        val expectedFreeEveningMs = dayStartMs + (18 * 3600000L) + (30 * 60000L)
        assertTrue(
            "Task during commute buffer must be shifted to 18:30 or later, was ${formatTime(results[0].scheduledTimeMs)}",
            results[0].scheduledTimeMs >= expectedFreeEveningMs
        )
    }

    @Test
    fun testSleepWindowProtection() {
        val prefs = LifeRhythmPrefs().apply {
            sleepHour = 23 // 11:00 PM
        }
        val validator = SmartScheduleValidator(prefs)

        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        // Task proposed at 11:30 PM (23:30)
        val lateTask = CommitmentEntity(
            id = 3L,
            title = "Late Night Coding",
            category = CommitmentCategory.WORK,
            scheduledTimeMs = dayStartMs + (23 * 3600000L) + (30 * 60000L),
            durationMinutes = 60
        )

        val results = validator.validateAndRebalance(listOf(lateTask), dayStartMs)
        assertEquals(1, results.size)

        val calResult = Calendar.getInstance().apply { timeInMillis = results[0].scheduledTimeMs }
        val hour = calResult.get(Calendar.HOUR_OF_DAY)
        assertTrue(
            "Late night task must be shifted to morning hours (hour $hour was unexpected)",
            hour in 6..10
        )
    }

    @Test
    fun testConflictResolutionAndChaining() {
        val prefs = LifeRhythmPrefs().apply {
            collegeStartHour = 9
            collegeEndHour = 18
            transitBufferMinutes = 30
        }
        val validator = SmartScheduleValidator(prefs)

        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis
        val eveningStart = dayStartMs + (18 * 3600000L) + (30 * 60000L) // 18:30

        // Two tasks scheduled at the exact same time (18:30)
        val task1 = CommitmentEntity(
            id = 11L,
            title = "Workout",
            category = CommitmentCategory.FITNESS,
            scheduledTimeMs = eveningStart,
            durationMinutes = 30
        )

        val task2 = CommitmentEntity(
            id = 12L,
            title = "OS Lab Assignment",
            category = CommitmentCategory.EDUCATION,
            scheduledTimeMs = eveningStart,
            durationMinutes = 45
        )

        val results = validator.validateAndRebalance(listOf(task1, task2), dayStartMs)
        assertEquals(2, results.size)

        val first = results[0]
        val second = results[1]

        val firstEnd = first.scheduledTimeMs + (first.durationMinutes * 60000L)
        assertTrue(
            "Second task start (${formatTime(second.scheduledTimeMs)}) must not overlap first task end (${formatTime(firstEnd)})",
            second.scheduledTimeMs >= firstEnd
        )
    }

    @Test
    fun testEarlyMorning1AmTaskShiftedToDaytime() {
        val prefs = LifeRhythmPrefs().apply {
            quietHoursStart = 22 // 10:00 PM
            quietHoursEnd = 7    // 07:00 AM
        }
        val validator = SmartScheduleValidator(prefs)

        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        // Task proposed at 1:00 AM (01:00) - exactly what caused the user's issue!
        val oneAmTask = CommitmentEntity(
            id = 99L,
            title = "Sleep Disruption Test Task",
            category = CommitmentCategory.GENERAL,
            scheduledTimeMs = dayStartMs + (1 * 3600000L), // 01:00 AM
            durationMinutes = 30
        )

        val results = validator.validateAndRebalance(listOf(oneAmTask), dayStartMs)
        assertEquals(1, results.size)

        val calResult = Calendar.getInstance().apply { timeInMillis = results[0].scheduledTimeMs }
        val hour = calResult.get(Calendar.HOUR_OF_DAY)
        val minute = calResult.get(Calendar.MINUTE)

        // Must NOT be at 1:00 AM; must be after quiet hours end (7:30 AM)
        assertTrue(
            "1:00 AM task must be shifted out of sleep window, was ${formatTime(results[0].scheduledTimeMs)}",
            hour >= 7
        )
        assertEquals(7, hour)
        assertEquals(30, minute)
    }

    @Test
    fun testQuietHoursEvaluation() {
        val prefs = LifeRhythmPrefs().apply {
            quietHoursStart = 22 // 10:00 PM
            quietHoursEnd = 7    // 07:00 AM
        }

        val cal = Calendar.getInstance()

        // 1:00 AM -> Quiet hours
        cal.set(Calendar.HOUR_OF_DAY, 1)
        assertTrue("1:00 AM must be inside quiet hours", prefs.isInsideQuietHours(cal.timeInMillis))

        // 11:30 PM (23:30) -> Quiet hours
        cal.set(Calendar.HOUR_OF_DAY, 23)
        cal.set(Calendar.MINUTE, 30)
        assertTrue("11:30 PM must be inside quiet hours", prefs.isInsideQuietHours(cal.timeInMillis))

        // 6:59 AM -> Quiet hours
        cal.set(Calendar.HOUR_OF_DAY, 6)
        cal.set(Calendar.MINUTE, 59)
        assertTrue("6:59 AM must be inside quiet hours", prefs.isInsideQuietHours(cal.timeInMillis))

        // 7:01 AM -> Daytime (NOT quiet hours)
        cal.set(Calendar.HOUR_OF_DAY, 7)
        cal.set(Calendar.MINUTE, 1)
        assertTrue("7:01 AM must be daytime", !prefs.isInsideQuietHours(cal.timeInMillis))

        // 2:00 PM (14:00) -> Daytime
        cal.set(Calendar.HOUR_OF_DAY, 14)
        assertTrue("2:00 PM must be daytime", !prefs.isInsideQuietHours(cal.timeInMillis))
    }

    @Test
    fun testMuteCallsForToday() {
        val prefs = LifeRhythmPrefs().apply {
            quietHoursEnd = 7
        }

        prefs.muteCallsForToday()
        assertTrue("isCallMutedNow must be true after muteCallsForToday", prefs.isCallMutedNow())

        prefs.unmuteCalls()
        // During normal daytime (e.g. 2 PM), unmute should make isCallMutedNow false
        val cal = Calendar.getInstance().apply { set(Calendar.HOUR_OF_DAY, 14) }
        assertTrue("isInsideQuietHours at 2 PM should be false", !prefs.isInsideQuietHours(cal.timeInMillis))
    }

    @Test
    fun testChecklistNotesPriorParsing() {
        val formattedNotes = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.buildChecklistNotes(
            brief = "Relay Manoj's message: 'Bro message to Ajay that I am not coming'.",
            messageQuote = "Bro message to Ajay that I am not coming",
            priority = "HIGH",
            sender = "Manoj"
        )

        val priority = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parsePriority(formattedNotes)
        val quote = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseMessageQuote(formattedNotes)

        assertEquals("HIGH", priority)
        assertEquals("Bro message to Ajay that I am not coming", quote)
    }

    @Test
    fun testWebDomainClassifierGroceries() {
        val result = com.zipper.app.webmind.data.WebDomainClassifier.classify("compare tomato")
        assertEquals(com.zipper.app.webmind.data.QueryDomain.QUICK_GROCERY, result.domain)
        assertTrue("Should be comparison query", result.isComparison)
        assertEquals("tomato", result.cleanEntity)

        val onionResult = com.zipper.app.webmind.data.WebDomainClassifier.classify("price of 1kg onion across zepto and blinkit")
        assertEquals(com.zipper.app.webmind.data.QueryDomain.QUICK_GROCERY, onionResult.domain)
        assertTrue(onionResult.cleanEntity.contains("onion"))
    }

    @Test
    fun testWebDomainClassifierTech() {
        val mobileResult = com.zipper.app.webmind.data.WebDomainClassifier.classify("compare mobile")
        assertEquals(com.zipper.app.webmind.data.QueryDomain.ELECTRONICS_TECH, mobileResult.domain)
        assertTrue(mobileResult.isComparison)

        val phoneResult = com.zipper.app.webmind.data.WebDomainClassifier.classify("compare iPhone 16 vs Samsung S24")
        assertEquals(com.zipper.app.webmind.data.QueryDomain.ELECTRONICS_TECH, phoneResult.domain)
        assertTrue(phoneResult.isComparison)
    }

    @Test
    fun testOfflineKnowledgeCatalogTomatoLookup() {
        val payload = com.zipper.app.webmind.data.OfflineKnowledgeCatalog.lookupOffline("compare tomato")
        assertNotNull("Tomato comparison must not be null in offline catalog", payload)
        assertTrue("Must contain Blinkit item", payload!!.items.any { it.store == "Blinkit" })
        assertTrue("Must contain Zepto item", payload.items.any { it.store == "Zepto" })
        assertEquals("Blinkit", payload.recommendedItem.store)
        assertTrue("Must be marked offline", payload.isOffline)
    }

    @Test
    fun testOfflineKnowledgeCatalogMobileLookup() {
        val payload = com.zipper.app.webmind.data.OfflineKnowledgeCatalog.lookupOffline("compare mobile")
        assertNotNull("Mobile comparison must not be null in offline catalog", payload)
        assertTrue("Must contain flagship smartphones", payload!!.items.size >= 2)
        assertTrue(payload.summaryMarkdown.contains("Galaxy S24"))
    }

    @Test
    fun testSamsungHealthWorkoutParsing() {
        val ocrSample = """
            Walking
            3.23 km
            Thu, 15 Oct, 2026 7:03 am
            Workout details
            Workout duration 25:36
            Avg. speed 7.5 km/h
            Workout calories 203 kcal
            Steps 3,507
            Avg. pace 07'54" /km
            Total duration 1:04:27
        """.trimIndent()

        val metrics = com.zipper.app.core.intelligence.health.HealthWorkoutIngestionEngine.parseWorkoutMetrics(
            title = "Walking",
            text = ocrSample
        )

        assertNotNull("Workout metrics should be successfully parsed", metrics)
        assertEquals("Walking", metrics!!.exerciseType)
        assertEquals(3.23, metrics.distanceKm ?: 0.0, 0.01)
        assertEquals(26, metrics.durationMinutes) // 25:36 rounded to 26 mins
        assertEquals(203, metrics.caloriesKcal)
        assertEquals(3507, metrics.steps)
        assertEquals(7.5, metrics.avgSpeedKmh ?: 0.0, 0.01)
        assertEquals("07:54/km", metrics.avgPaceMinKm)
        assertEquals("1:04:27", metrics.totalDurationFormatted)

        // Math verification checks
        assertNotNull(metrics.calculatedSpeedKmh)
        assertTrue("Calculated speed should be ~7.4-7.6 km/h", metrics.calculatedSpeedKmh!! in 7.3..7.7)
        assertNotNull(metrics.calculatedCadenceStepsPerKm)
        assertEquals(1086, metrics.calculatedCadenceStepsPerKm) // 3507 / 3.23 = 1085.75 -> 1086

        val markdownTable = com.zipper.app.core.intelligence.health.HealthWorkoutIngestionEngine.formatMarkdownVerificationTable(metrics)
        assertTrue(markdownTable.contains("3.23 km"))
        assertTrue(markdownTable.contains("203 kcal"))
        assertTrue(markdownTable.contains("3,507"))
        assertTrue(markdownTable.contains("Consistent") || markdownTable.contains("Close"))
    }

    @Test
    fun testAutoDoCategorizationFitnessNeverFakesWorkout() {
        val workoutItem = CommitmentEntity(
            id = 201L,
            title = "Morning Walk & Jog",
            category = CommitmentCategory.FITNESS,
            scheduledTimeMs = System.currentTimeMillis()
        )

        val plan = com.zipper.app.core.intelligence.planner.AutoDoPlanResolver.resolve(workoutItem)
        assertEquals(false, plan.canInstantSend)
        assertEquals("com.sec.android.app.shealth", plan.steps[0].packageName)
        assertEquals("Samsung Health", plan.steps[0].appName)
        assertTrue("Must explicitly mention real workout tracking without faking", plan.explanation.contains("Samsung Health"))
    }

    @Test
    fun testAutoDoEmailWithAttachmentResolution() {
        val emailItem = CommitmentEntity(
            id = 202L,
            title = "Send final update email",
            category = CommitmentCategory.COMMUNICATION,
            notes = "send email to recipient@example.com saying i have done all things and also attach your resume over there"
        )

        val plan = com.zipper.app.core.intelligence.planner.AutoDoPlanResolver.resolve(emailItem)
        assertTrue("Must be classified as email", plan.isEmail)
        assertEquals(1, plan.steps.size)
        assertEquals("recipient@example.com", plan.steps[0].recipient)
        assertEquals("com.google.android.gm", plan.steps[0].packageName)
        assertEquals("Gmail", plan.steps[0].appName)
        assertTrue("Must require attachment", plan.requiresAttachment)
        assertEquals("resume", plan.attachmentKeyword)
        assertTrue("Message must reflect instructions", plan.steps[0].message.contains("done all things", ignoreCase = true))
    }

    @Test
    fun testMultiPlatformCrawlTrackerMathProof() {
        val badges = listOf(
            com.zipper.app.webmind.data.PlatformCrawlBadge(
                storeName = "Blinkit",
                status = com.zipper.app.webmind.data.CrawlPlatformStatus.CRAWLED,
                productTitle = "Desi Tomato 1kg",
                price = 39.0,
                priceDisplay = "₹39/kg",
                deliveryTime = "12 mins",
                badge = "Lowest Price 🏆",
                targetUrl = "https://blinkit.com"
            ),
            com.zipper.app.webmind.data.PlatformCrawlBadge(
                storeName = "Zepto",
                status = com.zipper.app.webmind.data.CrawlPlatformStatus.CRAWLED,
                productTitle = "Hybrid Tomato 500g",
                price = 44.0,
                priceDisplay = "₹44/kg",
                deliveryTime = "10 mins",
                badge = "⚡ 10 mins",
                targetUrl = "https://zepto.com"
            )
        )

        val report = com.zipper.app.webmind.data.MultiPlatformCrawlTracker.buildReport("tomato", "Tomato", badges)
        assertEquals(39.0, report.lowestPrice, 0.01)
        assertEquals(44.0, report.highestPrice, 0.01)
        assertEquals(5.0, report.savingsAmount, 0.01)
        assertTrue("Savings % should be ~11.36%", report.savingsPercentage in 11.0..12.0)
        assertTrue(report.mathProof.contains("Saves ₹5"))
        assertTrue(report.verdictText.contains("Blinkit offers the best price"))
    }

    @Test
    fun testDeviceAssetResolverFormatting() {
        val kbFormatted = com.zipper.app.core.intelligence.resolver.DeviceAssetResolver.formatFileSize(1024L * 184)
        assertEquals("184 KB", kbFormatted)

        val mbFormatted = com.zipper.app.core.intelligence.resolver.DeviceAssetResolver.formatFileSize((1024L * 1024 * 1.5).toLong())
        assertEquals("1.5 MB", mbFormatted)

        val todayFormatted = com.zipper.app.core.intelligence.resolver.DeviceAssetResolver.formatLastModified(System.currentTimeMillis())
        assertTrue("Must start with Today", todayFormatted.startsWith("Today"))
    }

    private fun formatTime(timeMs: Long): String {
        val cal = Calendar.getInstance().apply { timeInMillis = timeMs }
        return String.format("%02d:%02d", cal.get(Calendar.HOUR_OF_DAY), cal.get(Calendar.MINUTE))
    }
}
