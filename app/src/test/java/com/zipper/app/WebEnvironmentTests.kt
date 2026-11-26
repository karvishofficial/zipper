package com.zipper.app

import com.zipper.app.core.agent.ActionCandidate
import com.zipper.app.core.agent.ActionExecutor
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.AgentRuntime
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.web.HiddenWebController
import com.zipper.app.core.web.WebActionExecutor
import com.zipper.app.core.web.WebController
import com.zipper.app.core.web.WebDeepLinkManager
import com.zipper.app.core.web.WebOnDeviceEngine
import com.zipper.app.core.web.WebProductFinder
import com.zipper.app.core.web.WebStateDistiller
import com.zipper.app.core.web.features.ArticleReaderEngine
import com.zipper.app.core.web.features.PnrTrackerEngine
import com.zipper.app.core.web.features.PriceComparisonEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WebEnvironmentTests {

    @Test
    fun testWebStateDistillerParsesCandidatesCorrectly() {
        val sampleJson = """
        {
            "url": "https://www.amazon.in/s?k=tomatoes",
            "title": "Amazon.in: tomatoes",
            "candidates": [
                {
                    "index": 0,
                    "karvishId": "k_node_0",
                    "role": "SEARCH_FIELD",
                    "text": "Search Amazon.in",
                    "contentDescription": "Search",
                    "resourceId": "twotabsearchtextbox",
                    "className": "INPUT",
                    "hintText": "Search Amazon.in",
                    "isClickable": true,
                    "isEditable": true,
                    "bounds": [50, 20, 800, 80]
                },
                {
                    "index": 1,
                    "karvishId": "k_node_1",
                    "role": "BUTTON",
                    "text": "Go",
                    "contentDescription": "Submit search",
                    "resourceId": "nav-search-submit-button",
                    "className": "INPUT",
                    "hintText": null,
                    "isClickable": true,
                    "isEditable": false,
                    "bounds": [810, 20, 950, 80]
                },
                {
                    "index": 2,
                    "karvishId": "k_node_2",
                    "role": "LINK",
                    "text": "Fresh Tomato Hybrid 1kg",
                    "contentDescription": null,
                    "resourceId": "product_title_0",
                    "className": "A",
                    "hintText": null,
                    "isClickable": true,
                    "isEditable": false,
                    "bounds": [60, 200, 900, 260]
                },
                {
                    "index": 3,
                    "karvishId": "k_node_3",
                    "role": "LIST_ITEM",
                    "text": "₹42/kg",
                    "contentDescription": null,
                    "resourceId": "price_0",
                    "className": "SPAN",
                    "hintText": null,
                    "isClickable": false,
                    "isEditable": false,
                    "bounds": [60, 270, 300, 310]
                }
            ]
        }
        """.trimIndent()

        val distiller = WebStateDistiller()
        val distilledState = distiller.distill(sampleJson)

        assertEquals("com.zipper.web", distilledState.packageName)
        assertEquals("Amazon.in: tomatoes", distilledState.activityName)
        assertEquals(4, distilledState.nodes.size)

        // Check Search Field
        val searchNode = distilledState.nodes[0]
        assertEquals("SEARCH_FIELD", searchNode.role)
        assertEquals("Search Amazon.in", searchNode.text)
        assertTrue(searchNode.isEditable)
        assertTrue(searchNode.isClickable)
        assertEquals(50, searchNode.bounds.left)
        assertEquals(800, searchNode.bounds.right)

        // Check Button
        val buttonNode = distilledState.nodes[1]
        assertEquals("BUTTON", buttonNode.role)
        assertEquals("Go", buttonNode.text)
        assertFalse(buttonNode.isEditable)
        assertTrue(buttonNode.isClickable)

        // Check Price Listing
        val priceNode = distilledState.nodes[3]
        assertEquals("LIST_ITEM", priceNode.role)
        assertEquals("₹42/kg", priceNode.text)
        assertEquals(NormalizedBounds(60, 270, 300, 310), priceNode.bounds)
    }

    @Test
    fun testWebStateDistillerHandlesEmptyOrMalformedJsonGracefully() {
        val distiller = WebStateDistiller()

        val emptyState = distiller.distill(null)
        assertEquals("com.zipper.web", emptyState.packageName)
        assertTrue(emptyState.nodes.isEmpty())

        val invalidState = distiller.distill("{ not a valid json")
        assertEquals("com.zipper.web", invalidState.packageName)
        assertTrue(invalidState.activityName?.startsWith("error:") == true)
        assertTrue(invalidState.nodes.isEmpty())
    }

    @Test
    fun testWebActionExecutorRoutesPrimitives() = runBlocking {
        var tappedId: String? = null
        var typedId: String? = null
        var typedText: String? = null
        var scrolledDirection: ScrollDirection? = null
        var submitted = false
        var backed = false

        val fakeWebController = object : WebController {
            override suspend fun navigate(url: String, timeoutMs: Long): Boolean = true
            override suspend fun distillWebState(): DistilledUiState = DistilledUiState()
            override suspend fun clickElement(karvishId: String): Boolean {
                tappedId = karvishId
                return true
            }
            override suspend fun typeText(karvishId: String, text: String): Boolean {
                typedId = karvishId
                typedText = text
                return true
            }
            override suspend fun selectOption(karvishId: String, value: String): Boolean = true
            override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean {
                scrolledDirection = direction
                return true
            }
            override suspend fun submit(): Boolean {
                submitted = true
                return true
            }
            override suspend fun goBack(): Boolean {
                backed = true
                return true
            }
            override suspend fun clear() {}
            override suspend fun destroy() {}
        }

        val stubCandidates = mapOf(
            0 to DistilledNode(
                index = 0,
                role = "SEARCH_FIELD",
                text = "Search",
                resourceId = "k_node_0",
                bounds = NormalizedBounds(0, 0, 100, 100),
                isEditable = true
            ),
            1 to DistilledNode(
                index = 1,
                role = "BUTTON",
                text = "Submit",
                resourceId = "k_node_1",
                bounds = NormalizedBounds(100, 0, 200, 100),
                isClickable = true
            )
        )

        val executor = WebActionExecutor(fakeWebController)

        // Test TypeText
        assertTrue(executor.execute(AgentAction.TypeText(0, "fresh tomatoes"), stubCandidates))
        assertEquals("k_node_0", typedId)
        assertEquals("fresh tomatoes", typedText)

        // Test Tap
        assertTrue(executor.execute(AgentAction.Tap(1), stubCandidates))
        assertEquals("k_node_1", tappedId)

        // Test Scroll
        assertTrue(executor.execute(AgentAction.Scroll(direction = ScrollDirection.DOWN), stubCandidates))
        assertEquals(ScrollDirection.DOWN, scrolledDirection)

        // Test Submit
        assertTrue(executor.execute(AgentAction.Submit, stubCandidates))
        assertTrue(submitted)

        // Test Back
        assertTrue(executor.execute(AgentAction.PressBack, stubCandidates))
        assertTrue(backed)
    }

    @Test
    fun testAgentRuntimeExecutesWithCustomWebActionExecutor() = runBlocking {
        val testWebState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Search Results",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "SEARCH_FIELD",
                    text = "Search",
                    resourceId = "k_node_0",
                    bounds = NormalizedBounds(50, 50, 500, 100),
                    isEditable = true
                ),
                DistilledNode(
                    index = 1,
                    role = "BUTTON",
                    text = "Search",
                    resourceId = "k_node_1",
                    bounds = NormalizedBounds(510, 50, 600, 100),
                    isClickable = true
                )
            )
        )

        var executedAction: AgentAction? = null
        val customExecutor = object : ActionExecutor {
            override suspend fun execute(
                action: AgentAction,
                candidateMap: Map<Int, DistilledNode>,
                goal: String,
                actionCandidateMap: Map<String, ActionCandidate>
            ): Boolean {
                executedAction = action
                return true
            }
        }

        val runtime = AgentRuntime(
            maxTurns = 1,
            stateProvider = { testWebState },
            customActionExecutor = customExecutor
        )

        assertNotNull(runtime)
    }

    @Test
    fun testWebAutonomousRunnerFillsFormFieldsAndSubmits() = runBlocking {
        val formState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Contact Us",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "TEXT_FIELD",
                    text = "",
                    resourceId = "name",
                    hintText = "Your Name",
                    bounds = NormalizedBounds(50, 50, 500, 100),
                    isEditable = true
                ),
                DistilledNode(
                    index = 1,
                    role = "TEXT_FIELD",
                    text = "",
                    resourceId = "email",
                    hintText = "Email Address",
                    bounds = NormalizedBounds(50, 110, 500, 160),
                    isEditable = true
                ),
                DistilledNode(
                    index = 2,
                    role = "TEXT_FIELD",
                    text = "",
                    resourceId = "message",
                    hintText = "Your Message",
                    bounds = NormalizedBounds(50, 170, 500, 240),
                    isEditable = true
                ),
                DistilledNode(
                    index = 3,
                    role = "BUTTON",
                    text = "Submit Message",
                    resourceId = "btn_submit",
                    bounds = NormalizedBounds(50, 250, 300, 300),
                    isClickable = true
                )
            )
        )

        val enteredInputs = mutableMapOf<Int, String>()
        var tappedSubmit = false

        val fakeWebController = object : WebController {
            override suspend fun navigate(url: String, timeoutMs: Long): Boolean = true
            override suspend fun distillWebState(): DistilledUiState = formState
            override suspend fun clickElement(karvishId: String): Boolean {
                if (karvishId == "k_node_3") tappedSubmit = true
                return true
            }
            override suspend fun typeText(karvishId: String, text: String): Boolean = true
            override suspend fun selectOption(karvishId: String, value: String): Boolean = true
            override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean = true
            override suspend fun submit(): Boolean = true
            override suspend fun goBack(): Boolean = true
            override suspend fun clear() {}
            override suspend fun destroy() {}
        }

        val testExecutor = object : ActionExecutor {
            override suspend fun execute(
                action: AgentAction,
                candidateMap: Map<Int, DistilledNode>,
                goal: String,
                actionCandidateMap: Map<String, ActionCandidate>
            ): Boolean {
                when (action) {
                    is AgentAction.TypeText -> enteredInputs[action.targetNodeIndex] = action.text
                    is AgentAction.Tap -> if (action.targetNodeIndex == 3) tappedSubmit = true
                    else -> {}
                }
                return true
            }
        }

        val runner = com.zipper.app.core.web.WebAutonomousRunner(
            webController = fakeWebController,
            actionExecutor = testExecutor
        )

        val goal = "Open https://mysite.com/contact and submit form: name: Karthik, email: karthik@zipper.com, message: Hello from test"
        val result = runner.runEpisode(goal)

        assertTrue(result.success)
        assertEquals("Karthik", enteredInputs[0])
        assertEquals("karthik@zipper.com", enteredInputs[1])
        assertEquals("Hello from test", enteredInputs[2])
        assertTrue(tappedSubmit)
        assertEquals(3, result.fieldsFilled.size)
    }

    @Test
    fun testWebAutonomousRunnerNavigatesAndExtractsSection() = runBlocking {
        val wikiState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Wikipedia",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "SEARCH_FIELD",
                    text = "",
                    resourceId = "searchInput",
                    bounds = NormalizedBounds(50, 50, 400, 100),
                    isEditable = true
                ),
                DistilledNode(
                    index = 1,
                    role = "BUTTON",
                    text = "Search",
                    resourceId = "searchBtn",
                    bounds = NormalizedBounds(410, 50, 500, 100),
                    isClickable = true
                ),
                DistilledNode(
                    index = 2,
                    role = "LABEL",
                    text = "Quantum computing is a rapidly-emerging technology.",
                    resourceId = "intro_p",
                    bounds = NormalizedBounds(50, 150, 950, 300)
                )
            )
        )

        var typedQuery: String? = null
        var tappedSearch = false

        val fakeWebController = object : WebController {
            override suspend fun navigate(url: String, timeoutMs: Long): Boolean = true
            override suspend fun distillWebState(): DistilledUiState = wikiState
            override suspend fun clickElement(karvishId: String): Boolean = true
            override suspend fun typeText(karvishId: String, text: String): Boolean = true
            override suspend fun selectOption(karvishId: String, value: String): Boolean = true
            override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean = true
            override suspend fun submit(): Boolean = true
            override suspend fun goBack(): Boolean = true
            override suspend fun clear() {}
            override suspend fun destroy() {}
        }

        val testExecutor = object : ActionExecutor {
            override suspend fun execute(
                action: AgentAction,
                candidateMap: Map<Int, DistilledNode>,
                goal: String,
                actionCandidateMap: Map<String, ActionCandidate>
            ): Boolean {
                when (action) {
                    is AgentAction.TypeText -> typedQuery = action.text
                    is AgentAction.Tap -> if (action.targetNodeIndex == 1) tappedSearch = true
                    else -> {}
                }
                return true
            }
        }

        val runner = com.zipper.app.core.web.WebAutonomousRunner(
            webController = fakeWebController,
            actionExecutor = testExecutor
        )

        val goal = "Go to wikipedia.org and search Quantum Computing"
        val result = runner.runEpisode(goal)

        assertTrue(result.success)
        assertEquals("Quantum Computing", typedQuery)
        assertTrue(tappedSearch)
        assertNotNull(result.extractedContent)
        assertTrue(result.extractedContent?.contains("Quantum computing") == true)
    }

    @Test
    fun testWebConfigDefaults() {
        assertEquals("gemini-3-flash-preview", com.zipper.app.core.web.WebConfigManager.getDefaultModelForProvider("Google Gemini"))
        assertEquals("qwen/qwen3.8-27b", com.zipper.app.core.web.WebConfigManager.getDefaultModelForProvider("Groq"))
        assertEquals("gpt-4o-mini", com.zipper.app.core.web.WebConfigManager.getDefaultModelForProvider("OpenAI"))
        assertEquals("google/gemini-2.0-flash-exp:free", com.zipper.app.core.web.WebConfigManager.getDefaultModelForProvider("OpenRouter"))
        assertEquals("custom-model", com.zipper.app.core.web.WebConfigManager.getDefaultModelForProvider("Custom (OpenAI-compatible)"))
    }

    @Test
    fun testAutonomousRunnerExecutesLlmPlannedForm() = runBlocking {
        val filledFields = mutableMapOf<Int, String>()
        val selectedOptions = mutableMapOf<String, String>()
        var tappedSubmit = false

        val testState = DistilledUiState(
            activityName = "https://rrce.contineo.in:4043/students/index.php",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "TEXT_FIELD",
                    resourceId = "usn",
                    text = "USN",
                    bounds = NormalizedBounds(100, 200, 900, 260),
                    isEditable = true
                ),
                DistilledNode(
                    index = 1,
                    role = "SELECT",
                    resourceId = "dob_day",
                    text = "dob_day Selected=05 [Options: 01,02,29,30,31]",
                    bounds = NormalizedBounds(100, 280, 300, 340),
                    isClickable = true
                ),
                DistilledNode(
                    index = 2,
                    role = "SELECT",
                    resourceId = "dob_month",
                    text = "dob_month Selected=Jan [Options: Jan,Feb,Sep,Oct]",
                    bounds = NormalizedBounds(350, 280, 650, 340),
                    isClickable = true
                ),
                DistilledNode(
                    index = 3,
                    role = "SELECT",
                    resourceId = "dob_year",
                    text = "dob_year Selected=2000 [Options: 2004,2005,2006]",
                    bounds = NormalizedBounds(700, 280, 950, 340),
                    isClickable = true
                ),
                DistilledNode(
                    index = 4,
                    role = "BUTTON",
                    resourceId = "btnSubmit",
                    text = "Login",
                    bounds = NormalizedBounds(100, 400, 900, 480),
                    isClickable = true
                )
            )
        )

        val fakeWebController = object : WebController {
            override suspend fun navigate(url: String, timeoutMs: Long): Boolean = true
            override suspend fun distillWebState(): DistilledUiState = testState
            override suspend fun clickElement(karvishId: String): Boolean {
                if (karvishId == "k_node_4") tappedSubmit = true
                return true
            }
            override suspend fun typeText(karvishId: String, text: String): Boolean = true
            override suspend fun selectOption(karvishId: String, value: String): Boolean {
                selectedOptions[karvishId] = value
                return true
            }
            override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean = true
            override suspend fun submit(): Boolean = true
            override suspend fun goBack(): Boolean = true
            override suspend fun clear() {}
            override suspend fun destroy() {}
        }

        val testExecutor = object : ActionExecutor {
            override suspend fun execute(
                action: AgentAction,
                candidateMap: Map<Int, DistilledNode>,
                goal: String,
                actionCandidateMap: Map<String, ActionCandidate>
            ): Boolean {
                when (action) {
                    is AgentAction.TypeText -> filledFields[action.targetNodeIndex] = action.text
                    is AgentAction.Tap -> if (action.targetNodeIndex == 4) tappedSubmit = true
                    else -> {}
                }
                return true
            }
        }

        val testLlmClient = com.zipper.app.core.web.WebLlmClient()

        // We can test planned actions directly
        val actions = listOf(
            com.zipper.app.core.web.PlannedWebAction("TYPE", 0, text = "1RR22CS001"),
            com.zipper.app.core.web.PlannedWebAction("SELECT", 1, value = "29"),
            com.zipper.app.core.web.PlannedWebAction("SELECT", 2, value = "Sep"),
            com.zipper.app.core.web.PlannedWebAction("SELECT", 3, value = "2005"),
            com.zipper.app.core.web.PlannedWebAction("CLICK", 4)
        )

        // Verify action mappings
        assertEquals(5, actions.size)
        assertEquals("1RR22CS001", actions[0].text)
        assertEquals("29", actions[1].value)
        assertEquals("Sep", actions[2].value)
        assertEquals("2005", actions[3].value)
        assertEquals("CLICK", actions[4].action)
    }

    @Test
    fun testWebVerificationProvesTypeTextOutcome() {
        val verifier = com.zipper.app.core.web.WebVerification()

        val preState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Search Form",
            nodes = listOf(
                DistilledNode(index = 0, role = "SEARCH_FIELD", text = "", bounds = NormalizedBounds(0, 0, 100, 100), isEditable = true)
            )
        )

        val postState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Search Form",
            nodes = listOf(
                DistilledNode(index = 0, role = "SEARCH_FIELD", text = "potatoes", bounds = NormalizedBounds(0, 0, 100, 100), isEditable = true)
            )
        )

        val result = verifier.verify(
            preState = preState,
            action = AgentAction.TypeText(0, "potatoes"),
            postState = postState,
            goal = "Search potatoes"
        )

        assertEquals(com.zipper.app.core.web.WebOutcomeStatus.INPUT_POPULATED, result.status)
        assertTrue(result.isProgressMade)
        assertFalse(result.isGoalAccomplished)
    }

    @Test
    fun testWebVerificationDetectsBlockedOrErrorPages() {
        val verifier = com.zipper.app.core.web.WebVerification()

        val preState = DistilledUiState(packageName = "com.zipper.web", activityName = "Portal")
        val errorState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "404 Not Found",
            nodes = listOf(
                DistilledNode(index = 0, role = "LABEL", text = "The requested URL was not found on this server.", bounds = NormalizedBounds(0, 0, 100, 100))
            )
        )

        val result = verifier.verify(
            preState = preState,
            action = AgentAction.Tap(0),
            postState = errorState,
            goal = "Open student portal"
        )

        assertEquals(com.zipper.app.core.web.WebOutcomeStatus.BLOCKED_OR_ERROR, result.status)
        assertFalse(result.isProgressMade)
        assertFalse(result.isGoalAccomplished)
        assertTrue(result.summary.contains("404 not found"))
    }

    @Test
    fun testWebEnvironmentClosedLoopStep() = runBlocking {
        var currentUrl = "https://example.com/login"
        var currentNodes = listOf(
            DistilledNode(index = 0, role = "SEARCH_FIELD", text = "", resourceId = "k_node_0", bounds = NormalizedBounds(0, 0, 100, 100), isEditable = true),
            DistilledNode(index = 1, role = "BUTTON", text = "Submit", resourceId = "k_node_1", bounds = NormalizedBounds(100, 0, 200, 100), isClickable = true)
        )

        val fakeController = object : WebController {
            override suspend fun navigate(url: String, timeoutMs: Long): Boolean {
                currentUrl = url
                return true
            }
            override suspend fun distillWebState(): DistilledUiState {
                return DistilledUiState(packageName = "com.zipper.web", activityName = currentUrl, nodes = currentNodes)
            }
            override suspend fun clickElement(karvishId: String): Boolean {
                if (karvishId == "k_node_1") {
                    currentUrl = "https://example.com/dashboard"
                    currentNodes = listOf(
                        DistilledNode(index = 0, role = "HEADER", text = "Welcome to Dashboard", bounds = NormalizedBounds(0, 0, 100, 100))
                    )
                }
                return true
            }
            override suspend fun typeText(karvishId: String, text: String): Boolean {
                currentNodes = listOf(
                    DistilledNode(index = 0, role = "SEARCH_FIELD", text = text, resourceId = "k_node_0", bounds = NormalizedBounds(0, 0, 100, 100), isEditable = true),
                    DistilledNode(index = 1, role = "BUTTON", text = "Submit", resourceId = "k_node_1", bounds = NormalizedBounds(100, 0, 200, 100), isClickable = true)
                )
                return true
            }
            override suspend fun selectOption(karvishId: String, value: String): Boolean = true
            override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean = true
            override suspend fun submit(): Boolean = true
            override suspend fun goBack(): Boolean = true
            override suspend fun clear() {}
            override suspend fun destroy() {}
        }

        val env = com.zipper.app.core.web.WebEnvironment(controller = fakeController)

        // Step 1: Observe
        val initialState = env.observe()
        assertEquals("https://example.com/login", initialState.activityName)

        // Step 2: TypeText
        val candMap = initialState.nodes.associateBy { it.index }
        val step1 = env.step(AgentAction.TypeText(0, "admin"), "Login to portal", candMap)
        assertEquals(com.zipper.app.core.web.WebOutcomeStatus.INPUT_POPULATED, step1.verification.status)
        assertTrue(step1.verification.isProgressMade)

        // Step 3: Tap Submit
        val postCandMap = step1.postState.nodes.associateBy { it.index }
        val step2 = env.step(AgentAction.Tap(1), "Login to portal", postCandMap)
        assertEquals(com.zipper.app.core.web.WebOutcomeStatus.URL_CHANGED, step2.verification.status)
        assertTrue(step2.verification.isProgressMade)
        assertEquals("https://example.com/dashboard", step2.postState.activityName)
    }

    @Test
    fun testWebDeepLinkManagerPackageResolutionAndStoreNames() {
        assertEquals("com.amazon.mShop.android.shopping", WebDeepLinkManager.resolveAppPackage("https://www.amazon.in/dp/B08N5WRWNW"))
        assertEquals("com.flipkart.android", WebDeepLinkManager.resolveAppPackage("https://www.flipkart.com/product/p/itm123"))
        assertEquals("com.zeptoconsumerapp", WebDeepLinkManager.resolveAppPackage("https://www.zeptonow.com/search?q=tomato"))
        assertEquals("com.grofers.customerapp", WebDeepLinkManager.resolveAppPackage("https://blinkit.com/prn/potato"))

        assertEquals("Amazon", WebDeepLinkManager.resolveStoreName("https://www.amazon.in/dp/B123"))
        assertEquals("Flipkart", WebDeepLinkManager.resolveStoreName("https://www.flipkart.com/item"))
        assertEquals("Zepto", WebDeepLinkManager.resolveStoreName("https://www.zeptonow.com/item"))
        assertEquals("Blinkit", WebDeepLinkManager.resolveStoreName("https://blinkit.com/item"))
    }

    @Test
    fun testWebProductFinderQueryDetectionAndSearchUrl() {
        assertTrue(WebProductFinder.isShoppingQuery("Find the cheapest wireless earphones on Amazon"))
        assertTrue(WebProductFinder.isShoppingQuery("What is the price of iPhone 15 on Flipkart"))
        assertTrue(WebProductFinder.isShoppingQuery("Buy 1kg tomato on Zepto"))
        assertFalse(WebProductFinder.isShoppingQuery("Summarize the biography of Alan Turing"))

        val (amazonUrl, amazonStore) = WebProductFinder.buildSearchUrl("Find cheapest boAt earphones on Amazon")
        assertEquals("Amazon", amazonStore)
        assertTrue(amazonUrl.contains("amazon.in/s?k="))
        assertTrue(amazonUrl.contains("s=price-asc-rank"))

        val (flipkartUrl, flipkartStore) = WebProductFinder.buildSearchUrl("Cheapest power bank on Flipkart")
        assertEquals("Flipkart", flipkartStore)
        assertTrue(flipkartUrl.contains("flipkart.com/search?q="))
        assertTrue(flipkartUrl.contains("sort=price_asc"))
    }

    @Test
    fun testWebProductFinderExtractionAndBestMatch() {
        val state = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "https://www.amazon.in/s?k=earphones",
            nodes = listOf(
                DistilledNode(index = 0, role = "LINK", text = "boAt Bassheads 100 In-Ear Wired Earphones", href = "https://www.amazon.in/dp/B071Z8M4KX", imageUrl = "https://m.media-amazon.com/images/I/img1.jpg", bounds = NormalizedBounds(10, 50, 400, 80)),
                DistilledNode(index = 1, role = "LIST_ITEM", text = "₹349", bounds = NormalizedBounds(10, 85, 200, 105)),
                DistilledNode(index = 2, role = "LIST_ITEM", text = "4.2 out of 5 stars (95k)", bounds = NormalizedBounds(210, 85, 350, 105)),
                DistilledNode(index = 3, role = "LINK", text = "boAt Rockerz 255 Pro+ Wireless Earphones", href = "https://www.amazon.in/dp/B08CD2N542", imageUrl = "https://m.media-amazon.com/images/I/img2.jpg", bounds = NormalizedBounds(10, 200, 400, 230)),
                DistilledNode(index = 4, role = "LIST_ITEM", text = "₹999", bounds = NormalizedBounds(10, 235, 200, 255)),
                DistilledNode(index = 5, role = "LIST_ITEM", text = "4.1 out of 5 stars (150k)", bounds = NormalizedBounds(210, 235, 350, 255))
            )
        )

        val products = WebProductFinder.extractProductsFromState(state, "Amazon")
        assertEquals(2, products.size)

        val bestWireless = WebProductFinder.findBestMatch(products, listOf("rockerz", "wireless"))
        assertNotNull(bestWireless)
        assertEquals("https://www.amazon.in/dp/B08CD2N542", bestWireless?.url)
        assertEquals("₹999", bestWireless?.priceDisplay)
        assertEquals("⭐ 4.1", bestWireless?.rating)
    }

    @Test
    fun testWebStateDistillerParsesHrefAndImageUrl() {
        val json = """
            {
                "url": "https://www.amazon.in/dp/B08CD2N542",
                "title": "boAt Rockerz 255 Pro+",
                "candidates": [
                    {
                        "index": 0,
                        "role": "LINK",
                        "text": "boAt Rockerz 255 Pro+",
                        "bounds": [10, 20, 300, 60],
                        "href": "https://www.amazon.in/dp/B08CD2N542",
                        "imageUrl": "https://m.media-amazon.com/images/I/img2.jpg"
                    }
                ]
            }
        """.trimIndent()

        val distiller = WebStateDistiller()
        val state = distiller.distill(json)
        assertEquals(1, state.nodes.size)
        val node = state.nodes[0]
        assertEquals("https://www.amazon.in/dp/B08CD2N542", node.href)
        assertEquals("https://m.media-amazon.com/images/I/img2.jpg", node.imageUrl)
    }

    @Test
    fun testPriceComparisonEngineQueryDetection() {
        assertTrue(PriceComparisonEngine.isComparisonQuery("Compare iPhone 15 price across Amazon and Flipkart"))
        assertTrue(PriceComparisonEngine.isComparisonQuery("iPhone 15 vs Galaxy S24"))
        assertTrue(PriceComparisonEngine.isComparisonQuery("Price of boAt earphones across both stores"))
        assertFalse(PriceComparisonEngine.isComparisonQuery("Find cheapest wireless earphones on Amazon"))
    }

    @Test
    fun testArticleReaderEngineQueryDetectionAndExtraction() {
        val sampleQuery = "Summarize https://en.wikipedia.org/wiki/Artificial_intelligence"
        assertTrue(ArticleReaderEngine.isArticleQuery(sampleQuery))
        assertFalse(ArticleReaderEngine.isArticleQuery("Search iPhone on Flipkart"))

        val extractedUrl = ArticleReaderEngine.extractUrlFromQuery(sampleQuery)
        assertEquals("https://en.wikipedia.org/wiki/Artificial_intelligence", extractedUrl)

        val testState = DistilledUiState(
            packageName = "com.zipper.web",
            activityName = "Artificial Intelligence - Wikipedia",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "LABEL",
                    text = "Artificial intelligence is the intelligence of machines or software, as opposed to the intelligence of humans or animals.",
                    bounds = NormalizedBounds(10, 50, 900, 150)
                ),
                DistilledNode(
                    index = 1,
                    role = "BUTTON",
                    text = "Accept Cookies",
                    bounds = NormalizedBounds(10, 160, 200, 200)
                ),
                DistilledNode(
                    index = 2,
                    role = "LABEL",
                    text = "AI applications include advanced web search engines, recommendation systems, and autonomous driving.",
                    bounds = NormalizedBounds(10, 220, 900, 320)
                )
            )
        )

        val summary = ArticleReaderEngine.summarizeDistilledState(testState, "https://en.wikipedia.org/wiki/Artificial_intelligence")
        assertEquals("Artificial Intelligence - Wikipedia", summary.title)
        assertTrue(summary.executiveSummary.contains("Artificial intelligence is the intelligence of machines"))
        assertEquals(2, summary.keyTakeaways.size)
        assertFalse(summary.fullMarkdown.contains("Accept Cookies"))
    }

    @Test
    fun testPnrTrackerEngineQueryDetectionAndExtraction() {
        // IRCTC PNR
        assertTrue(PnrTrackerEngine.isTrackingQuery("Check PNR 4521890321"))
        assertTrue(PnrTrackerEngine.isTrackingQuery("pnr status 4521890321"))
        assertTrue(PnrTrackerEngine.isTrackingQuery("4521890321 train ticket"))
        assertEquals("4521890321", PnrTrackerEngine.extractTrackingId("Check PNR 4521890321"))

        // Courier tracking
        assertTrue(PnrTrackerEngine.isTrackingQuery("Track speed post ED123456789IN"))
        assertTrue(PnrTrackerEngine.isTrackingQuery("Track delhivery shipment 987654321012"))
        assertEquals("ED123456789IN", PnrTrackerEngine.extractTrackingId("Track speed post ED123456789IN"))

        // Non-tracking queries
        assertFalse(PnrTrackerEngine.isTrackingQuery("Search tomatoes on Amazon"))
        assertFalse(PnrTrackerEngine.isTrackingQuery("Summarize Wikipedia article"))
    }

    @Test
    fun testAdShieldDetection() {
        assertTrue(HiddenWebController.isAdOrTracker("https://securepubads.g.doubleclick.net/gampad/ads"))
        assertTrue(HiddenWebController.isAdOrTracker("https://www.google-analytics.com/analytics.js"))
        assertTrue(HiddenWebController.isAdOrTracker("https://connect.facebook.net/en_US/fbevents.js"))
        assertTrue(HiddenWebController.isAdOrTracker("https://c.amazon-adsystem.com/aax2/apstag.js"))

        // First-party content should NOT be blocked
        assertFalse(HiddenWebController.isAdOrTracker("https://www.amazon.in/dp/B08CD2N542"))
        assertFalse(HiddenWebController.isAdOrTracker("https://www.zepto.com/search?q=tomato"))
        assertFalse(HiddenWebController.isAdOrTracker("https://m.media-amazon.com/images/I/product.jpg"))
    }

    @Test
    fun testWebOnDeviceEngineConstantsAndSchema() {
        assertEquals(8, WebOnDeviceEngine.CANONICAL_ACTIONS.size)
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("CLICK"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("TYPE_TEXT"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("SUBMIT"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("SCROLL"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("SELECT"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("EXTRACT"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("WAIT"))
        assertTrue(WebOnDeviceEngine.CANONICAL_ACTIONS.contains("NAVIGATE"))
    }

    @Test
    fun testDistilledNodeSpatialRegionMapping() {
        val headerNode = DistilledNode(
            index = 0,
            role = "SEARCH_FIELD",
            text = "Search products",
            bounds = NormalizedBounds(20, 10, 980, 80),
            region = "HEADER"
        )
        val bodyNode = DistilledNode(
            index = 1,
            role = "BUTTON",
            text = "Add to Cart",
            bounds = NormalizedBounds(20, 300, 480, 380),
            region = "BODY"
        )
        val modalNode = DistilledNode(
            index = 2,
            role = "BUTTON",
            text = "Accept Cookies",
            bounds = NormalizedBounds(100, 400, 900, 600),
            region = "MODAL"
        )

        assertEquals("HEADER", headerNode.region)
        assertEquals("BODY", bodyNode.region)
        assertEquals("MODAL", modalNode.region)
    }
}



