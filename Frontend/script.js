/* =========================================================
   AGRICONNECT FRONTEND
   Connects to the provided FastAPI backend
========================================================= */


/* =========================================================
   API CONFIGURATION
========================================================= */

const API_BASE_URL = "http://127.0.0.1:8000";


/* =========================================================
   APPLICATION STATE
========================================================= */

let products = [];

let cart = JSON.parse(
    localStorage.getItem("agriconnect_cart") || "[]"
);

let currentCategory = "All";

let searchTerm = "";

let sortOption = "default";


/* =========================================================
   PRODUCT EMOJI MAPPING
========================================================= */

const productEmoji = {

    "Fresh Tomatoes": "🍅",
    "Organic Potatoes": "🥔",
    "Fresh Mangoes": "🥭",
    "Basmati Rice": "🍚",
    "Red Onions": "🧅",
    "Fresh Milk": "🥛",
    "Organic Wheat": "🌾",
    "Green Chillies": "🌶️"

};


/* =========================================================
   DOM HELPERS
========================================================= */

const $ = (selector) => document.querySelector(selector);

const $$ = (selector) =>
    document.querySelectorAll(selector);


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    initializeTheme();

    initializeNavigation();

    initializeMarketplace();

    initializeCart();

    initializeModals();

    initializeForms();

    initializeAITools();

    initializeDashboard();

    initializeOrders();

    checkAPI();

    loadProducts();

    updateCartUI();

    setTimeout(() => {

        $("#loadingScreen").classList.add("hide");

    }, 700);

});


/* =========================================================
   API REQUEST HELPER
========================================================= */

async function apiRequest(
    endpoint,
    options = {}
) {

    const response = await fetch(
        `${API_BASE_URL}${endpoint}`,
        {
            ...options,

            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            }
        }
    );


    let data = null;

    try {
        data = await response.json();
    }

    catch {
        data = null;
    }


    if (!response.ok) {

        const message =
            data?.detail ||
            `Request failed with status ${response.status}`;

        throw new Error(message);
    }


    return data;
}


/* =========================================================
   API HEALTH CHECK
========================================================= */

async function checkAPI() {

    const dot = $("#apiStatusDot");

    const text = $("#apiStatusText");

    const wrapper = document.querySelector(".api-status");


    try {

        const result = await apiRequest("/");

        if (result.status === "ok") {

            wrapper.classList.add("online");

            text.textContent = "API Online";
        }

    }

    catch (error) {

        wrapper.classList.add("offline");

        text.textContent = "API Offline";

        console.error(error);
    }
}


/* =========================================================
   PRODUCTS
========================================================= */

async function loadProducts() {

    const loading = $("#productsLoading");

    const grid = $("#productsGrid");

    const empty = $("#emptyProducts");


    loading.classList.remove("hidden");

    grid.innerHTML = "";

    empty.classList.add("hidden");


    try {

        products = await apiRequest("/products");

        $("#heroProductCount").textContent =
            `${products.length}+`;

        $("#dashboardProducts").textContent =
            products.length;

        renderProducts();

    }

    catch (error) {

        grid.innerHTML = `
            <div class="empty-state" style="grid-column:1/-1;">
                <div class="empty-icon">⚠️</div>

                <h3>Unable to load products</h3>

                <p>
                    Make sure your FastAPI server is running at
                    <strong>127.0.0.1:8000</strong>.
                </p>

                <button
                    class="btn btn-primary"
                    onclick="loadProducts()"
                >
                    Try Again
                </button>
            </div>
        `;

        showToast(
            "Could not connect to the FastAPI backend.",
            "error"
        );
    }

    finally {

        loading.classList.add("hidden");
    }
}


/* =========================================================
   PRODUCT FILTERING
========================================================= */

function getFilteredProducts() {

    let result = [...products];


    if (currentCategory !== "All") {

        result = result.filter(
            product =>
                product.category === currentCategory
        );
    }


    if (searchTerm.trim()) {

        const query =
            searchTerm.trim().toLowerCase();

        result = result.filter(product => {

            return (

                product.name
                    .toLowerCase()
                    .includes(query)

                ||

                product.category
                    .toLowerCase()
                    .includes(query)

                ||

                product.farmer
                    .toLowerCase()
                    .includes(query)

                ||

                product.location
                    .toLowerCase()
                    .includes(query)

            );

        });
    }


    switch (sortOption) {

        case "price-low":

            result.sort(
                (a,b) => a.price - b.price
            );

            break;


        case "price-high":

            result.sort(
                (a,b) => b.price - a.price
            );

            break;


        case "rating":

            result.sort(
                (a,b) => b.rating - a.rating
            );

            break;


        case "name":

            result.sort(
                (a,b) =>
                    a.name.localeCompare(b.name)
            );

            break;
    }


    return result;
}


/* =========================================================
   RENDER PRODUCTS
========================================================= */

function renderProducts() {

    const grid = $("#productsGrid");

    const empty = $("#emptyProducts");

    const filtered = getFilteredProducts();


    grid.innerHTML = "";


    if (!filtered.length) {

        empty.classList.remove("hidden");

        return;
    }


    empty.classList.add("hidden");


    filtered.forEach(product => {

        grid.appendChild(
            createProductCard(product)
        );

    });
}


/* =========================================================
   PRODUCT CARD
========================================================= */

function createProductCard(product) {

    const card = document.createElement("article");

    card.className = "product-card";


    const emoji =
        productEmoji[product.name] || "🌱";


    card.innerHTML = `

        <div class="product-image">

            <span class="product-badge">
                ${escapeHTML(product.badge || "Farm Fresh")}
            </span>

            <span class="product-rating">
                ★ ${Number(product.rating).toFixed(1)}
            </span>

            <span>
                ${emoji}
            </span>

        </div>


        <div class="product-info">

            <span class="product-category">
                ${escapeHTML(product.category)}
            </span>

            <h3 class="product-name">
                ${escapeHTML(product.name)}
            </h3>

            <p class="product-farmer">
                👨‍🌾 ${escapeHTML(product.farmer)}
                <br>
                📍 ${escapeHTML(product.location)}
            </p>


            <div class="product-bottom">

                <div class="product-price">
                    ₹${Number(product.price).toFixed(0)}
                    <small>/ ${escapeHTML(product.unit)}</small>
                </div>


                <button
                    class="add-cart-btn"
                    data-add-cart="${product.id}"
                    title="Add to cart"
                >
                    🛒
                </button>

            </div>


            <button
                class="view-product-btn"
                data-view-product="${product.id}"
            >
                View product details
            </button>

        </div>
    `;


    return card;
}


/* =========================================================
   MARKETPLACE INITIALIZATION
========================================================= */

function initializeMarketplace() {


    $("#searchInput")
        .addEventListener("input", event => {

            searchTerm = event.target.value;

            $("#clearSearchBtn").style.display =
                searchTerm ? "block" : "none";

            renderProducts();
        });


    $("#clearSearchBtn")
        .addEventListener("click", () => {

            $("#searchInput").value = "";

            searchTerm = "";

            $("#clearSearchBtn").style.display =
                "none";

            renderProducts();
        });


    $$(".category-btn")
        .forEach(button => {

            button.addEventListener("click", () => {

                $$(".category-btn")
                    .forEach(btn =>
                        btn.classList.remove("active")
                    );

                button.classList.add("active");

                currentCategory =
                    button.dataset.category;

                renderProducts();
            });

        });


    $("#sortSelect")
        .addEventListener("change", event => {

            sortOption = event.target.value;

            renderProducts();
        });


    $("#refreshProductsBtn")
        .addEventListener(
            "click",
            loadProducts
        );


    $("#resetFiltersBtn")
        .addEventListener("click", resetFilters);


    $("#productsGrid")
        .addEventListener("click", event => {

            const cartButton =
                event.target.closest("[data-add-cart]");

            const viewButton =
                event.target.closest("[data-view-product]");


            if (cartButton) {

                addToCart(
                    Number(cartButton.dataset.addCart)
                );

                return;
            }


            if (viewButton) {

                openProductModal(
                    Number(viewButton.dataset.viewProduct)
                );
            }

        });
}


/* =========================================================
   RESET FILTERS
========================================================= */

function resetFilters() {

    searchTerm = "";

    currentCategory = "All";

    sortOption = "default";


    $("#searchInput").value = "";

    $("#sortSelect").value = "default";


    $$(".category-btn")
        .forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.category === "All"
            );

        });


    $("#clearSearchBtn").style.display =
        "none";


    renderProducts();
}


/* =========================================================
   CART
========================================================= */

function initializeCart() {

    $("#cartBtn")
        .addEventListener(
            "click",
            openCart
        );


    $("#closeCartBtn")
        .addEventListener(
            "click",
            closeCart
        );


    $("#cartOverlay")
        .addEventListener(
            "click",
            closeCart
        );


    $("#clearCartBtn")
        .addEventListener(
            "click",
            clearCart
        );


    $("#checkoutBtn")
        .addEventListener(
            "click",
            openCheckout
        );


    $("#cartItems")
        .addEventListener("click", event => {

            const increase =
                event.target.closest(
                    "[data-cart-increase]"
                );

            const decrease =
                event.target.closest(
                    "[data-cart-decrease]"
                );

            const remove =
                event.target.closest(
                    "[data-cart-remove]"
                );


            if (increase) {

                changeCartQuantity(
                    Number(increase.dataset.cartIncrease),
                    1
                );
            }


            if (decrease) {

                changeCartQuantity(
                    Number(decrease.dataset.cartDecrease),
                    -1
                );
            }


            if (remove) {

                removeFromCart(
                    Number(remove.dataset.cartRemove)
                );
            }

        });
}


/* =========================================================
   ADD TO CART
========================================================= */

function addToCart(productId) {

    const product =
        products.find(
            item => item.id === productId
        );


    if (!product) {

        showToast(
            "Product not found.",
            "error"
        );

        return;
    }


    const existing =
        cart.find(
            item => item.product_id === productId
        );


    if (existing) {

        existing.quantity += 1;

    }

    else {

        cart.push({
            product_id: productId,
            quantity: 1
        });

    }


    saveCart();

    updateCartUI();


    showToast(
        `${product.name} added to cart 🛒`
    );
}


/* =========================================================
   CHANGE CART QUANTITY
========================================================= */

function changeCartQuantity(
    productId,
    amount
) {

    const item =
        cart.find(
            item => item.product_id === productId
        );


    if (!item) {
        return;
    }


    item.quantity += amount;


    if (item.quantity <= 0) {

        cart =
            cart.filter(
                item =>
                    item.product_id !== productId
            );
    }


    saveCart();

    updateCartUI();
}


/* =========================================================
   REMOVE FROM CART
========================================================= */

function removeFromCart(productId) {

    cart =
        cart.filter(
            item =>
                item.product_id !== productId
        );


    saveCart();

    updateCartUI();
}


/* =========================================================
   CLEAR CART
========================================================= */

function clearCart() {

    if (!cart.length) {
        return;
    }


    cart = [];

    saveCart();

    updateCartUI();


    showToast(
        "Cart cleared.",
        "warning"
    );
}


/* =========================================================
   SAVE CART
========================================================= */

function saveCart() {

    localStorage.setItem(
        "agriconnect_cart",
        JSON.stringify(cart)
    );
}


/* =========================================================
   CART TOTAL
========================================================= */

function calculateCartTotal() {

    return cart.reduce(
        (total, item) => {

            const product =
                products.find(
                    p =>
                        p.id === item.product_id
                );


            if (!product) {
                return total;
            }


            return total +
                product.price *
                item.quantity;

        },
        0
    );
}


/* =========================================================
   CART ITEM COUNT
========================================================= */

function calculateCartCount() {

    return cart.reduce(
        (total, item) =>
            total + item.quantity,
        0
    );
}


/* =========================================================
   UPDATE CART UI
========================================================= */

function updateCartUI() {

    $("#cartCount").textContent =
        calculateCartCount();


    $("#cartTotal").textContent =
        formatCurrency(
            calculateCartTotal()
        );


    $("#checkoutTotal").textContent =
        formatCurrency(
            calculateCartTotal()
        );


    renderCartItems();
}


/* =========================================================
   RENDER CART
========================================================= */

function renderCartItems() {

    const container =
        $("#cartItems");


    if (!cart.length) {

        container.innerHTML = `

            <div class="cart-empty">

                <div>🛒</div>

                <h3>Your cart is empty</h3>

                <p>
                    Add some fresh produce to get started.
                </p>

            </div>

        `;

        return;
    }


    container.innerHTML = "";


    cart.forEach(item => {

        const product =
            products.find(
                p => p.id === item.product_id
            );


        if (!product) {
            return;
        }


        const emoji =
            productEmoji[product.name] || "🌱";


        const itemElement =
            document.createElement("div");


        itemElement.className =
            "cart-item";


        itemElement.innerHTML = `

            <div class="cart-item-image">
                ${emoji}
            </div>


            <div>

                <div class="cart-item-name">
                    ${escapeHTML(product.name)}
                </div>

                <div class="cart-item-price">
                    ₹${product.price}
                    / ${escapeHTML(product.unit)}
                </div>


                <div class="quantity-control">

                    <button
                        data-cart-decrease="${product.id}"
                    >
                        −
                    </button>

                    <span>
                        ${item.quantity}
                    </span>

                    <button
                        data-cart-increase="${product.id}"
                    >
                        +
                    </button>

                    <button
                        data-cart-remove="${product.id}"
                        title="Remove"
                    >
                        🗑️
                    </button>

                </div>

            </div>


            <div class="cart-item-total">

                ${formatCurrency(
                    product.price *
                    item.quantity
                )}

            </div>

        `;


        container.appendChild(itemElement);

    });
}


/* =========================================================
   OPEN CART
========================================================= */

function openCart() {

    $("#cartDrawer").classList.add("active");

    $("#cartOverlay").classList.add("active");

    document.body.classList.add("no-scroll");
}


/* =========================================================
   CLOSE CART
========================================================= */

function closeCart() {

    $("#cartDrawer").classList.remove("active");

    $("#cartOverlay").classList.remove("active");

    document.body.classList.remove("no-scroll");
}


/* =========================================================
   CHECKOUT
========================================================= */

function openCheckout() {

    if (!cart.length) {

        showToast(
            "Your cart is empty.",
            "warning"
        );

        return;
    }


    closeCart();

    openModal("checkoutModal");
}


/* =========================================================
   CHECKOUT FORM
========================================================= */

function initializeForms() {


    $("#checkoutForm")
        .addEventListener(
            "submit",
            submitOrder
        );


}


/* =========================================================
   PLACE ORDER
========================================================= */

async function submitOrder(event) {

    event.preventDefault();


    if (!cart.length) {

        showToast(
            "Your cart is empty.",
            "warning"
        );

        return;
    }


    const button =
        $("#placeOrderBtn");


    const form =
        new FormData(event.target);


    const orderData = {

        customer_name:
            form.get("customer_name").trim(),

        phone:
            form.get("phone").trim(),

        address:
            form.get("address").trim(),

        payment_method:
            form.get("payment_method"),

        items:
            cart.map(item => ({
                product_id: item.product_id,
                quantity: item.quantity
            }))

    };


    setButtonLoading(
        button,
        "Placing Order..."
    );


    try {

        const order =
            await apiRequest(
                "/orders",
                {
                    method: "POST",
                    body: JSON.stringify(orderData)
                }
            );


        showToast(
            `Order #${order.id} placed successfully! 🎉`
        );


        cart = [];

        saveCart();

        updateCartUI();


        event.target.reset();


        closeModal("checkoutModal");


        await loadOrders();

        await refreshDashboard();


        document
            .querySelector("#orders")
            .scrollIntoView({
                behavior: "smooth"
            });

    }

    catch (error) {

        showToast(
            error.message,
            "error"
        );

    }

    finally {

        resetButton(
            button,
            "Place Order"
        );
    }
}


/* =========================================================
   FARMER REGISTRATION

   Farmer registration now lives on farmer-register.html.
========================================================= */


/* =========================================================
   LOAD ORDERS
========================================================= */

async function loadOrders() {

    const container =
        $("#ordersList");


    container.innerHTML = `

        <div class="orders-placeholder">

            <div class="loading-spinner dark"></div>

            <p>
                Loading orders...
            </p>

        </div>

    `;


    try {

        const orders =
            await apiRequest("/orders");


        $("#dashboardOrders").textContent =
            orders.length;


        if (!orders.length) {

            container.innerHTML = `

                <div class="orders-placeholder">

                    <div>📦</div>

                    <h3>No orders yet</h3>

                    <p>
                        Your completed orders will appear here.
                    </p>

                </div>

            `;

            return;
        }


        container.innerHTML = "";


        orders
            .slice()
            .reverse()
            .forEach(order => {

                const card =
                    document.createElement("div");

                card.className = "order-card";


                card.innerHTML = `

                    <div>

                        <div class="order-id">
                            Order #${order.id}
                        </div>

                        <div class="order-date">
                            ${formatDate(order.created_at)}
                            <br>
                            ${escapeHTML(order.customer_name)}
                        </div>

                    </div>


                    <div class="order-total">
                        ${formatCurrency(order.total)}
                    </div>


                    <div class="order-status">
                        ${escapeHTML(order.status)}
                    </div>

                `;


                container.appendChild(card);

            });

    }

    catch (error) {

        container.innerHTML = `

            <div class="orders-placeholder">

                <div>⚠️</div>

                <h3>Could not load orders</h3>

                <p>
                    ${escapeHTML(error.message)}
                </p>

            </div>

        `;
    }
}


/* =========================================================
   DASHBOARD
========================================================= */

function initializeDashboard() {

    $("#refreshDashboardBtn")
        .addEventListener(
            "click",
            refreshDashboard
        );

}


async function refreshDashboard() {

    try {

        const [
            productData,
            orderData,
            farmerData
        ] = await Promise.all([

            apiRequest("/products"),

            apiRequest("/orders"),

            apiRequest("/farmers")

        ]);


        $("#dashboardProducts").textContent =
            productData.length;


        $("#dashboardOrders").textContent =
            orderData.length;


        $("#dashboardFarmers").textContent =
            farmerData.length;

    }

    catch (error) {

        console.error(
            "Dashboard refresh failed:",
            error
        );
    }
}


/* =========================================================
   ORDERS INITIALIZATION
========================================================= */

function initializeOrders() {

    $("#refreshOrdersBtn")
        .addEventListener(
            "click",
            loadOrders
        );


    loadOrders();

}


/* =========================================================
   AI TOOLS
========================================================= */

let forecastPairs = [];


function initializeAITools() {

    $("#forecastBtn")
        .addEventListener(
            "click",
            generateForecast
        );

    $("#forecastRegion")
        .addEventListener(
            "change",
            updateForecastCrops
        );

    loadForecastOptions();


    $("#optimizeRouteBtn")
        .addEventListener(
            "click",
            optimizeRoute
        );
}


/* =========================================================
   LOAD REAL FORECAST OPTIONS
========================================================= */

async function loadForecastOptions() {

    const regionSelect = $("#forecastRegion");
    const cropSelect = $("#forecastCrop");

    if (!regionSelect || !cropSelect) {
        return;
    }

    try {

        const data =
            await apiRequest("/forecast/options");

        forecastPairs = data.pairs || [];

        regionSelect.innerHTML = `
            <option value="">Select region</option>
        `;

        data.regions.forEach(region => {

            const option =
                document.createElement("option");

            option.value = region;
            option.textContent = region;

            regionSelect.appendChild(option);
        });

        if (data.regions.length) {
            regionSelect.value = data.regions[0];
            updateForecastCrops();
        }

    }
    catch (error) {

        regionSelect.innerHTML = `
            <option value="">Could not load regions</option>
        `;

        cropSelect.innerHTML = `
            <option value="">Could not load crops</option>
        `;

        console.error(
            "Forecast options failed:",
            error
        );
    }
}


function updateForecastCrops() {

    const region = $("#forecastRegion").value;
    const cropSelect = $("#forecastCrop");

    if (!cropSelect) {
        return;
    }

    const crops = forecastPairs
        .filter(pair => pair.region === region)
        .map(pair => pair.crop);

    cropSelect.innerHTML = `
        <option value="">Select crop</option>
    `;

    crops.forEach(crop => {

        const option =
            document.createElement("option");

        option.value = crop;
        option.textContent = crop;

        cropSelect.appendChild(option);
    });

    if (crops.length) {
        cropSelect.value = crops[0];
    }
}


/* =========================================================
   REAL ML DEMAND FORECAST
========================================================= */

async function generateForecast() {

    const button =
        $("#forecastBtn");

    const result =
        $("#forecastResult");

    const region =
        $("#forecastRegion").value;

    const crop =
        $("#forecastCrop").value;

    if (!region || !crop) {

        showToast(
            "Please select a region and crop.",
            "warning"
        );

        return;
    }

    setButtonLoading(
        button,
        "Analyzing..."
    );

    try {

        const endpoint =
            `/forecast?region=${encodeURIComponent(region)}&crop=${encodeURIComponent(crop)}`;

        const data =
            await apiRequest(endpoint);

        result.innerHTML = `

            <div class="forecast-main">

                <div>

                    <div class="forecast-crop">
                        Predicted Demand · ${escapeHTML(data.forecast_month)}
                    </div>

                    <div class="forecast-value">
                        ${Number(data.expected_demand).toLocaleString("en-IN", { maximumFractionDigits: 2 })}
                        <small>units</small>
                    </div>

                    <strong>
                        ${escapeHTML(data.crop)}
                    </strong>

                    <div class="forecast-month">
                        📍 ${escapeHTML(data.region)}
                    </div>

                    <div class="forecast-records">
                        Based on ${data.historical_records} historical observations
                    </div>

                </div>

                <div
                    class="confidence"
                    style="--confidence:${data.confidence}%"
                    title="Forecast reliability based on available historical data"
                >
                    <span>
                        ${data.confidence}%
                    </span>
                </div>

            </div>

            <div class="recommendation">
                💡 ${escapeHTML(data.recommendation)}
            </div>

        `;

        showToast(
            "Real AI demand forecast generated 🤖"
        );

    }
    catch (error) {

        result.innerHTML = `

            <div class="forecast-placeholder">

                <div>⚠️</div>

                <p>
                    ${escapeHTML(error.message)}
                </p>

            </div>

        `;

        showToast(
            error.message,
            "error"
        );

    }
    finally {

        resetButton(
            button,
            "✨ Generate Forecast"
        );
    }
}


/* =========================================================
   ROUTE OPTIMIZATION
========================================================= */

async function optimizeRoute() {

    const button =
        $("#optimizeRouteBtn");


    const origin =
        $("#routeOrigin")
            .value
            .trim();


    const destinationText =
        $("#routeDestinations")
            .value
            .trim();


    if (!origin) {

        showToast(
            "Please enter a starting point.",
            "warning"
        );

        return;
    }


    if (!destinationText) {

        showToast(
            "Please enter at least one destination.",
            "warning"
        );

        return;
    }


    const destinations =
        destinationText
            .split(",")
            .map(item => item.trim())
            .filter(Boolean);


    if (!destinations.length) {

        showToast(
            "Please enter valid destinations.",
            "warning"
        );

        return;
    }


    setButtonLoading(
        button,
        "Optimizing..."
    );


    try {

        const result =
            await apiRequest(
                "/optimize-route",
                {
                    method: "POST",

                    body: JSON.stringify({
                        origin,
                        destinations
                    })
                }
            );


        const container =
            $("#routeResult");


        container.classList.remove(
            "hidden"
        );


        container.innerHTML = `

            <div class="route-stats">

                <div class="route-stat">

                    <strong>
                        ${result.distance} km
                    </strong>

                    <span>
                        Distance
                    </span>

                </div>


                <div class="route-stat">

                    <strong>
                        ${result.time} min
                    </strong>

                    <span>
                        Est. Time
                    </span>

                </div>


                <div class="route-stat">

                    <strong>
                        ${result.fuel_saved}%
                    </strong>

                    <span>
                        Fuel Saved
                    </span>

                </div>

            </div>


            <div class="route-path">
                🏁 ${escapeHTML(result.route)}
            </div>

        `;


        showToast(
            "Route optimized successfully 🚚"
        );

    }

    catch (error) {

        showToast(
            error.message,
            "error"
        );

    }

    finally {

        resetButton(
            button,
            "🚀 Optimize Route"
        );
    }
}


/* =========================================================
   PRODUCT DETAIL MODAL
========================================================= */

function openProductModal(productId) {

    const product =
        products.find(
            p => p.id === productId
        );


    if (!product) {
        return;
    }


    const emoji =
        productEmoji[product.name] || "🌱";


    $("#productDetail").innerHTML = `

        <div class="product-detail-image">
            ${emoji}
        </div>


        <h2>
            ${escapeHTML(product.name)}
        </h2>


        <div class="detail-meta">

            <span class="detail-tag">
                ${escapeHTML(product.category)}
            </span>

            <span class="detail-tag">
                ★ ${Number(product.rating).toFixed(1)}
            </span>

            <span class="detail-tag">
                ${escapeHTML(product.badge || "Farm Fresh")}
            </span>

        </div>


        <div class="detail-price">
            ₹${Number(product.price).toFixed(0)}
            <small>
                / ${escapeHTML(product.unit)}
            </small>
        </div>


        <p>
            Fresh ${escapeHTML(product.name)}
            directly from
            <strong>
                ${escapeHTML(product.farmer)}
            </strong>.
        </p>


        <p class="detail-location">
            📍 ${escapeHTML(product.location)}
        </p>


        <button
            class="btn btn-primary full-width"
            id="detailAddCartBtn"
        >
            🛒 Add to Cart
        </button>

    `;


    $("#detailAddCartBtn")
        .addEventListener(
            "click",
            () => {

                addToCart(product.id);

                closeModal("productModal");

            }
        );


    openModal("productModal");
}


/* =========================================================
   MODALS
========================================================= */

function initializeModals() {

    $$("[data-close-modal]")
        .forEach(button => {

            button.addEventListener(
                "click",
                () => {

                    closeModal(
                        button.dataset.closeModal
                    );

                }
            );

        });


    $$(".modal-container")
        .forEach(container => {

            container.addEventListener(
                "click",
                event => {

                    if (
                        event.target === container
                    ) {

                        closeModal(
                            container.id
                        );

                    }

                }
            );

        });


    document.addEventListener(
        "keydown",
        event => {

            if (event.key === "Escape") {

                $$(".modal-container.active")
                    .forEach(modal => {

                        closeModal(modal.id);

                    });


                closeCart();
            }

        }
    );
}


function openModal(id) {

    const modal =
        document.getElementById(id);


    if (!modal) {
        return;
    }


    modal.classList.add("active");

    document.body.classList.add("no-scroll");
}


function closeModal(id) {

    const modal =
        document.getElementById(id);


    if (!modal) {
        return;
    }


    modal.classList.remove("active");


    if (
        !document.querySelector(
            ".modal-container.active"
        )
        &&
        !$("#cartDrawer").classList.contains(
            "active"
        )
    ) {

        document.body.classList.remove(
            "no-scroll"
        );
    }
}


/* =========================================================
   NAVIGATION
========================================================= */

function initializeNavigation() {

    $("#mobileMenuBtn")
        .addEventListener(
            "click",
            () => {

                $("#mainNav")
                    .classList.toggle("open");

            }
        );


    $$(".nav-link")
        .forEach(link => {

            link.addEventListener(
                "click",
                () => {

                    $("#mainNav")
                        .classList.remove("open");

                }
            );

        });


    window.addEventListener(
        "scroll",
        updateActiveNavigation
    );
}


function updateActiveNavigation() {

    const sections =
        document.querySelectorAll(
            "main section[id]"
        );


    let current = "home";


    sections.forEach(section => {

        const top =
            section.offsetTop - 130;


        if (
            window.scrollY >= top
        ) {

            current = section.id;

        }

    });


    $$(".nav-link")
        .forEach(link => {

            link.classList.toggle(
                "active",
                link.getAttribute("href") ===
                `#${current}`
            );

        });
}


/* =========================================================
   THEME
========================================================= */

function initializeTheme() {

    const savedTheme =
        localStorage.getItem(
            "agriconnect_theme"
        );


    if (savedTheme === "dark") {

        document.body.classList.add("dark");

        $("#themeBtn").textContent = "☀️";

    }


    $("#themeBtn")
        .addEventListener(
            "click",
            toggleTheme
        );
}


function toggleTheme() {

    const isDark =
        document.body.classList.toggle(
            "dark"
        );


    localStorage.setItem(
        "agriconnect_theme",
        isDark ? "dark" : "light"
    );


    $("#themeBtn").textContent =
        isDark ? "☀️" : "🌙";
}


/* =========================================================
   TOAST
========================================================= */

function showToast(
    message,
    type = "success"
) {

    const container =
        $("#toastContainer");


    const toast =
        document.createElement("div");


    toast.className =
        `toast ${type}`;


    toast.textContent =
        message;


    container.appendChild(toast);


    setTimeout(() => {

        toast.remove();

    }, 3600);
}


/* =========================================================
   BUTTON LOADING
========================================================= */

function setButtonLoading(
    button,
    text
) {

    if (!button) {
        return;
    }


    button.dataset.originalText =
        button.innerHTML;


    button.disabled = true;


    button.innerHTML = `

        <span
            class="loading-spinner"
            style="
                width:16px;
                height:16px;
                border-width:2px;
            "
        ></span>

        ${text}

    `;
}


function resetButton(
    button,
    fallbackText
) {

    if (!button) {
        return;
    }


    button.disabled = false;


    button.innerHTML =
        button.dataset.originalText ||
        fallbackText;
}


/* =========================================================
   FORMATTERS
========================================================= */

function formatCurrency(value) {

    return new Intl.NumberFormat(
        "en-IN",
        {
            style: "currency",
            currency: "INR",
            maximumFractionDigits: 2
        }
    ).format(value);
}


function formatDate(dateString) {

    if (!dateString) {
        return "Date unavailable";
    }


    const date =
        new Date(dateString);


    if (Number.isNaN(date.getTime())) {
        return dateString;
    }


    return date.toLocaleString(
        "en-IN",
        {
            dateStyle: "medium",
            timeStyle: "short"
        }
    );
}


/* =========================================================
   HTML ESCAPING
========================================================= */

function escapeHTML(value) {

    const div =
        document.createElement("div");


    div.textContent =
        String(value ?? "");


    return div.innerHTML;
}


/* =========================================================
   GLOBAL ERROR HANDLING
========================================================= */

window.addEventListener(
    "unhandledrejection",
    event => {

        console.error(
            "Unhandled promise rejection:",
            event.reason
        );

    }
);


/* =========================================================
   EXPORT FOR INLINE RETRIES
========================================================= */

window.loadProducts = loadProducts;