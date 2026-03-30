/**
 * SalonIQ Intelligence Platform -- Main Application
 *
 * Single-page application controller for the SalonIQ dashboard.
 * Handles navigation, API integration, chart rendering, and state management.
 */
(function () {
    'use strict';

    var API_BASE = '/api';

    /* ---- State ---- */

    var state = {
        currentView: 'dashboard',
        selectedSalonId: null,
        salons: [],
        isLoggedIn: false,
        atRiskCustomers: [],
        charts: {}
    };

    /* ---- Constants ---- */

    var VIEW_TITLES = {
        'dashboard': 'Dashboard',
        'at-risk': 'At-Risk Customers',
        'customer-detail': 'Customer Detail',
        'analytics': 'Analytics',
        'stylists': 'Stylist Performance',
        'monitoring': 'System Monitoring',
        'case-study': 'Case Study'
    };

    var RISK_COLORS = {
        'LOW': '#10B981',
        'MEDIUM': '#F59E0B',
        'HIGH': '#EF4444',
        'CRITICAL': '#DC2626'
    };

    var METRIC_NAMES = {
        churn_rate: 'Customer Churn Rate',
        avg_inter_visit_days: 'Avg Visit Interval',
        reengagement_rate: 'Re-engagement Rate',
        revenue_per_customer_year: 'Revenue per Customer/Year',
        admin_time_per_day: 'Admin Time per Day',
        customer_retention_12m: '12-Month Customer Retention',
        at_risk_identification_rate: 'At-Risk Identification Rate',
        campaign_response_rate: 'Campaign Response Rate',
        avg_ticket_value: 'Average Ticket Value',
        noshow_rate: 'No-Show Rate',
        active_customers: 'Active Customers',
        monthly_revenue: 'Monthly Revenue'
    };

    var METRIC_UNITS = {
        churn_rate: 'percent',
        avg_inter_visit_days: 'days',
        reengagement_rate: 'percent',
        revenue_per_customer_year: 'jpy',
        admin_time_per_day: 'minutes',
        customer_retention_12m: 'percent',
        at_risk_identification_rate: 'percent',
        campaign_response_rate: 'percent',
        avg_ticket_value: 'jpy',
        noshow_rate: 'percent',
        active_customers: 'count',
        monthly_revenue: 'jpy'
    };

    var LOWER_IS_BETTER = {
        churn_rate: true,
        avg_inter_visit_days: true,
        admin_time_per_day: true,
        noshow_rate: true
    };

    var JPY_FORMATTER = new Intl.NumberFormat('ja-JP', {
        style: 'currency',
        currency: 'JPY',
        maximumFractionDigits: 0
    });

    var NUMBER_FORMATTER = new Intl.NumberFormat('en-US');

    var debounceTimer = null;

    /* ================================================================
       Initialization
       ================================================================ */

    function setupLoginForm() {
        var form = document.getElementById('login-form');
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            var username = document.getElementById('login-username').value.trim();
            var password = document.getElementById('login-password').value.trim();
            var errorEl = document.getElementById('login-error');

            if (!username || !password) {
                errorEl.textContent = 'Please enter both username and password.';
                return;
            }

            errorEl.textContent = '';
            state.isLoggedIn = true;
            document.getElementById('login-screen').style.display = 'none';
            document.getElementById('app-main').style.display = 'flex';
            init();
        });
    }

    function init() {
        loadSalons();
        setupNavigation();
        setupEventListeners();

        // Handle browser back/forward navigation
        window.addEventListener('popstate', function () {
            var hash = window.location.hash.replace('#', '');
            if (hash && VIEW_TITLES[hash]) {
                navigateTo(hash);
            }
        });

        // Navigate to hash on initial load if present
        var initialHash = window.location.hash.replace('#', '');
        if (initialHash && VIEW_TITLES[initialHash]) {
            setTimeout(function () { navigateTo(initialHash); }, 100);
        }
    }

    function setupNavigation() {
        var links = document.querySelectorAll('.nav-link');
        links.forEach(function (link) {
            link.addEventListener('click', function (e) {
                e.preventDefault();
                var view = this.getAttribute('data-view');
                navigateTo(view);
            });
        });
    }

    function setupEventListeners() {
        var salonSelector = document.getElementById('salon-selector');
        salonSelector.addEventListener('change', function () {
            state.selectedSalonId = this.value;
            reloadCurrentView();
        });

        var riskFilter = document.getElementById('risk-filter');
        riskFilter.addEventListener('change', function () {
            loadAtRisk(this.value, document.getElementById('customer-search').value);
        });

        var searchInput = document.getElementById('customer-search');
        searchInput.addEventListener('input', function () {
            var value = this.value;
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(function () {
                filterAtRiskClientSide(value);
            }, 300);
        });

        var backBtn = document.getElementById('btn-back');
        backBtn.addEventListener('click', function () {
            navigateTo('at-risk');
        });

        var signOutLink = document.getElementById('sign-out-link');
        signOutLink.addEventListener('click', function (e) {
            e.preventDefault();
            state.isLoggedIn = false;
            state.currentView = 'dashboard';
            state.selectedSalonId = null;
            state.salons = [];
            state.atRiskCustomers = [];
            destroyAllCharts();
            document.getElementById('app-main').style.display = 'none';
            document.getElementById('login-screen').style.display = '';
        });

        var sidebarToggle = document.getElementById('sidebar-toggle');
        var sidebar = document.getElementById('sidebar');
        sidebarToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });
    }

    /* ================================================================
       Navigation
       ================================================================ */

    function navigateTo(view) {
        state.currentView = view;

        // Update browser URL hash
        if (window.location.hash !== '#' + view) {
            history.pushState(null, '', '#' + view);
        }

        document.querySelectorAll('.nav-link').forEach(function (link) {
            link.classList.toggle('active', link.getAttribute('data-view') === view);
        });

        document.querySelectorAll('.view').forEach(function (section) {
            section.classList.remove('active');
        });

        var viewEl = document.getElementById('view-' + view);
        if (viewEl) {
            viewEl.classList.add('active');
        }

        document.getElementById('page-title').textContent = VIEW_TITLES[view] || 'SalonIQ';

        var sidebar = document.getElementById('sidebar');
        sidebar.classList.remove('open');

        loadViewData(view);
    }

    function reloadCurrentView() {
        loadViewData(state.currentView);
    }

    function loadViewData(view) {
        if (!state.selectedSalonId && view !== 'monitoring') {
            return;
        }

        switch (view) {
            case 'dashboard':
                loadDashboard();
                break;
            case 'at-risk':
                loadAtRisk(
                    document.getElementById('risk-filter').value,
                    document.getElementById('customer-search').value
                );
                break;
            case 'analytics':
                loadAnalytics();
                break;
            case 'stylists':
                loadStylists();
                break;
            case 'monitoring':
                loadMonitoring();
                break;
            case 'case-study':
                loadCaseStudy();
                break;
        }
    }

    /* ================================================================
       API Layer
       ================================================================ */

    function fetchJSON(url) {
        return fetch(API_BASE + url)
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status + ': ' + response.statusText);
                }
                return response.json();
            });
    }

    /* ================================================================
       Salon Loading
       ================================================================ */

    function loadSalons() {
        fetchJSON('/v1/salons')
            .then(function (data) {
                state.salons = data.salons || [];
                var selector = document.getElementById('salon-selector');
                selector.innerHTML = '';

                state.salons.forEach(function (salon) {
                    var opt = document.createElement('option');
                    opt.value = salon.salon_id;
                    opt.textContent = salon.salon_name + ' (' + salon.city + ')';
                    selector.appendChild(opt);
                });

                if (state.salons.length > 0) {
                    state.selectedSalonId = state.salons[0].salon_id;
                    selector.value = state.selectedSalonId;
                    loadDashboard();
                }
            })
            .catch(function (err) {
                console.error('Failed to load salons:', err);
                var selector = document.getElementById('salon-selector');
                selector.innerHTML = '<option value="">Failed to load salons</option>';
            });
    }

    /* ================================================================
       Dashboard
       ================================================================ */

    function loadDashboard() {
        if (!state.selectedSalonId) return;

        var sid = state.selectedSalonId;

        showLoading('view-dashboard');

        var insightsPromise = fetchJSON('/v1/salon/' + sid + '/insights').catch(function () { return null; });
        var segmentsPromise = fetchJSON('/v1/analytics/segments/' + sid).catch(function () { return null; });

        Promise.all([
            fetchJSON('/v1/salon/' + sid + '/dashboard'),
            fetchJSON('/v1/salon/' + sid + '/at-risk?limit=5&min_risk=HIGH'),
            fetchJSON('/v1/analytics/revenue/' + sid),
            insightsPromise,
            segmentsPromise
        ])
            .then(function (results) {
                hideLoading('view-dashboard');
                var dashboard = results[0];
                var alerts = results[1];
                var revenue = results[2];
                var insights = results[3];
                var segments = results[4];

                renderDashboardKPIs(dashboard, insights);
                renderActionItems(insights);
                renderRiskDistributionChart(dashboard);
                renderDashboardSegmentsChart(segments);
                renderMiniRevenueChart(revenue);
                renderChurnDriversChart(dashboard);
                renderDashboardServicesChart(insights);
                renderBusinessInsights(insights);
                renderTopPerformer(insights);
                renderAlertsTable(alerts);
                renderRetentionTrendChart(insights);
            })
            .catch(function (err) {
                hideLoading('view-dashboard');
                showError('view-dashboard', 'Failed to load dashboard data. Ensure the API is running.');
                console.error('Dashboard load error:', err);
            });
    }

    function renderDashboardKPIs(data, insights) {
        var kpis = data.kpis || {};

        setTextContent('kpi-active-customers', formatNumber(kpis.total_active_customers || 0));
        setTextContent('kpi-retention', formatPercent(kpis.retention_rate_trailing_90d || 0));

        var atRiskEl = document.getElementById('kpi-at-risk');
        var pct = kpis.at_risk_percentage || 0;
        atRiskEl.textContent = formatPercent(pct);

        if (pct >= 30) {
            atRiskEl.style.color = RISK_COLORS.CRITICAL;
        } else if (pct >= 20) {
            atRiskEl.style.color = RISK_COLORS.HIGH;
        } else if (pct >= 10) {
            atRiskEl.style.color = RISK_COLORS.MEDIUM;
        } else {
            atRiskEl.style.color = RISK_COLORS.LOW;
        }

        if (insights) {
            var rev = insights.revenue || {};
            var visits = insights.visits || {};
            setTextContent('kpi-revenue-30d', formatCurrency(rev.last_30d || 0));
            setTextContent('kpi-avg-ticket', formatCurrency(rev.avg_ticket || kpis.avg_ticket_value || 0));

            var noshowRate = visits.noshow_rate || 0;
            setTextContent('kpi-noshow', (noshowRate * 100).toFixed(1) + '%');

            var trendRevenueEl = document.getElementById('kpi-trend-revenue');
            if (trendRevenueEl) {
                var momChange = rev.mom_change_pct || 0;
                if (momChange < 0) {
                    trendRevenueEl.className = 'kpi-trend negative';
                    trendRevenueEl.innerHTML = '&#9660; ' + Math.abs(momChange).toFixed(1) + '%';
                } else if (momChange > 0) {
                    trendRevenueEl.className = 'kpi-trend positive';
                    trendRevenueEl.innerHTML = '&#9650; ' + momChange.toFixed(1) + '%';
                } else {
                    trendRevenueEl.className = 'kpi-trend';
                    trendRevenueEl.textContent = '-- 0%';
                }
            }

            var trendRetentionEl = document.getElementById('kpi-trend-retention');
            if (trendRetentionEl) {
                var visitChange = visits.mom_change_pct || 0;
                if (visitChange < 0) {
                    trendRetentionEl.className = 'kpi-trend negative';
                    trendRetentionEl.innerHTML = '&#9660; ' + Math.abs(visitChange).toFixed(1) + '% visits';
                } else if (visitChange > 0) {
                    trendRetentionEl.className = 'kpi-trend positive';
                    trendRetentionEl.innerHTML = '&#9650; ' + visitChange.toFixed(1) + '% visits';
                } else {
                    trendRetentionEl.className = 'kpi-trend';
                    trendRetentionEl.textContent = '';
                }
            }
        } else {
            setTextContent('kpi-revenue-30d', formatCurrency(0));
            setTextContent('kpi-avg-ticket', formatCurrency(kpis.avg_ticket_value || 0));
            setTextContent('kpi-noshow', '--');
        }
    }

    function renderRiskDistributionChart(data) {
        var dist = data.risk_distribution || {};
        var labels = ['Low', 'Medium', 'High', 'Critical'];
        var values = [
            dist.LOW || 0,
            dist.MEDIUM || 0,
            dist.HIGH || 0,
            dist.CRITICAL || 0
        ];

        renderChart('chart-risk-dist', {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        RISK_COLORS.LOW,
                        RISK_COLORS.MEDIUM,
                        RISK_COLORS.HIGH,
                        RISK_COLORS.CRITICAL
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: { family: 'Inter', size: 12 }
                        }
                    }
                }
            }
        });
    }

    function renderChurnDriversChart(data) {
        var drivers = data.top_churn_drivers || [];
        var labels = drivers.map(function (d) { return d.factor || 'Unknown'; });
        var values = drivers.map(function (d) { return d.count || 0; });

        renderChart('chart-churn-drivers', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: values,
                    backgroundColor: '#D4A853',
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    function renderMiniRevenueChart(data) {
        var months = data.months || [];
        var labels = months.map(function (m) { return m.month; });
        var values = months.map(function (m) { return m.revenue; });

        renderChart('chart-mini-revenue', {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Revenue',
                    data: values,
                    borderColor: '#D4A853',
                    backgroundColor: 'rgba(212, 168, 83, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: '#D4A853',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return formatCurrencyShort(value);
                            }
                        }
                    }
                }
            }
        });
    }

    function renderAlertsTable(data) {
        var tbody = document.getElementById('alerts-tbody');
        var customers = data.at_risk_customers || [];

        if (customers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No at-risk alerts at this time.</td></tr>';
            return;
        }

        tbody.innerHTML = customers.slice(0, 5).map(function (c) {
            var badgeClass = getBadgeClass(c.risk_level);
            var overdue = (c.days_since_last_visit || 0) - (c.avg_inter_visit_days || 0);
            return '<tr>' +
                '<td><span class="badge ' + badgeClass + '">' + (c.risk_level || '--') + '</span></td>' +
                '<td>' + (c.customer_name || '--') + '</td>' +
                '<td>' + formatPercent(c.churn_probability || 0) + '</td>' +
                '<td>' + Math.max(0, Math.round(overdue)) + ' days</td>' +
                '</tr>';
        }).join('');
    }

    /* ================================================================
       Dashboard: New Render Functions
       ================================================================ */

    function renderActionItems(insights) {
        if (!insights) return;
        var actions = insights.action_items || {};
        var followupEl = document.querySelector('#action-followup .action-card-count');
        var bookingEl = document.querySelector('#action-booking .action-card-count');
        var offerEl = document.querySelector('#action-offer .action-card-count');

        if (followupEl) followupEl.textContent = actions.send_followup || 0;
        if (bookingEl) bookingEl.textContent = actions.book_appointment || 0;
        if (offerEl) offerEl.textContent = actions.offer_promotion || 0;
    }

    function renderDashboardSegmentsChart(data) {
        if (!data) return;
        var segments = data.segments || [];
        var labels = segments.map(function (s) { return s.name; });
        var values = segments.map(function (s) { return s.count; });
        var colors = segments.map(function (s) { return s.color || '#D4A853'; });

        renderChart('chart-dashboard-segments', {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: { family: 'Inter', size: 12 }
                        }
                    }
                }
            }
        });
    }

    function renderDashboardServicesChart(insights) {
        if (!insights) return;
        var serviceMix = insights.service_mix || [];
        var labels = serviceMix.map(function (s) { return s.category; });
        var values = serviceMix.map(function (s) { return s.revenue; });

        renderChart('chart-dashboard-services', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Revenue',
                    data: values,
                    backgroundColor: '#D4A853',
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return formatCurrencyShort(value);
                            }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    function renderBusinessInsights(insights) {
        var container = document.getElementById('dashboard-insights-list');
        if (!container) return;

        if (!insights || !insights.business_insights || insights.business_insights.length === 0) {
            container.innerHTML = '<div class="insight-item">No insights available at this time.</div>';
            return;
        }

        container.innerHTML = insights.business_insights.map(function (text) {
            return '<div class="insight-item">' + text + '</div>';
        }).join('');
    }

    function renderTopPerformer(insights) {
        if (!insights || !insights.top_stylist) return;

        var stylist = insights.top_stylist;
        setTextContent('performer-name', stylist.name || '--');
        setTextContent('performer-revenue', formatCurrency(stylist.revenue_30d || 0));
        setTextContent('performer-customers', stylist.customer_count || 0);
        setTextContent('performer-share', (stylist.revenue_share_pct || 0).toFixed(1) + '%');
    }

    function renderRetentionTrendChart(insights) {
        if (!insights || !insights.retention_trend || insights.retention_trend.length === 0) return;

        var trend = insights.retention_trend;
        var labels = trend.map(function (t) { return t.month; });
        var values = trend.map(function (t) { return t.rate; });

        renderChart('chart-retention-trend', {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Retention Rate',
                    data: values,
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.12)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: '#10B981',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    /* ================================================================
       At-Risk Customers
       ================================================================ */

    function loadAtRisk(minRisk, search) {
        if (!state.selectedSalonId) return;

        var url = '/v1/salon/' + state.selectedSalonId + '/at-risk?limit=100&min_risk=' + (minRisk || 'LOW');

        showLoading('view-at-risk');

        fetchJSON(url)
            .then(function (data) {
                hideLoading('view-at-risk');
                state.atRiskCustomers = data.at_risk_customers || [];

                renderAtRiskSummary(state.atRiskCustomers);

                var filtered = state.atRiskCustomers;
                if (search && search.trim()) {
                    var term = search.trim().toLowerCase();
                    filtered = filtered.filter(function (c) {
                        return (c.customer_name || '').toLowerCase().indexOf(term) >= 0;
                    });
                }

                renderAtRiskTable(filtered);
            })
            .catch(function (err) {
                hideLoading('view-at-risk');
                showError('view-at-risk', 'Failed to load at-risk customer data.');
                console.error('At-risk load error:', err);
            });
    }

    function filterAtRiskClientSide(search) {
        var filtered = state.atRiskCustomers;
        if (search && search.trim()) {
            var term = search.trim().toLowerCase();
            filtered = filtered.filter(function (c) {
                return (c.customer_name || '').toLowerCase().indexOf(term) >= 0;
            });
        }
        renderAtRiskTable(filtered);
    }

    function renderAtRiskTable(customers) {
        var tbody = document.getElementById('at-risk-tbody');
        var countLabel = document.getElementById('result-count');

        countLabel.textContent = customers.length + ' result' + (customers.length !== 1 ? 's' : '');

        if (customers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No at-risk customers matching the current filters.</td></tr>';
            return;
        }

        tbody.innerHTML = customers.map(function (c) {
            var badgeClass = getBadgeClass(c.risk_level);
            var action = c.suggested_action || 'FOLLOW_UP';
            var actionLabel = getActionLabel(action);
            var actionBtnClass = getActionBtnClass(action);
            var topFactor = c.primary_risk_factor || '--';

            return '<tr data-customer-id="' + (c.customer_id || '') + '">' +
                '<td>' + (c.customer_name || '--') + '</td>' +
                '<td><span class="badge ' + badgeClass + '">' + (c.risk_level || '--') + '</span></td>' +
                '<td>' + formatPercent(c.churn_probability || 0) + '</td>' +
                '<td>' + Math.round(c.days_since_last_visit || 0) + '</td>' +
                '<td>' + (c.avg_inter_visit_days || 0).toFixed(1) + '</td>' +
                '<td>' + formatCurrency(c.estimated_ltv_12m || 0) + '</td>' +
                '<td>' + topFactor + '</td>' +
                '<td><button class="btn btn-sm ' + actionBtnClass + '" onclick="event.stopPropagation()">' + actionLabel + '</button></td>' +
                '</tr>';
        }).join('');

        tbody.querySelectorAll('tr[data-customer-id]').forEach(function (row) {
            row.addEventListener('click', function () {
                var customerId = this.getAttribute('data-customer-id');
                if (customerId) {
                    loadCustomerDetail(customerId);
                }
            });
        });
    }

    function renderAtRiskSummary(customers) {
        var total = customers.length;
        var criticalCount = customers.filter(function (c) {
            return c.risk_level === 'CRITICAL';
        }).length;

        var avgChurn = 0;
        if (total > 0) {
            var sumChurn = customers.reduce(function (acc, c) {
                return acc + (c.churn_probability || 0);
            }, 0);
            avgChurn = (sumChurn / total) * 100;
        }

        var totalLtv = customers.reduce(function (acc, c) {
            return acc + (c.estimated_ltv_12m || 0);
        }, 0);

        setTextContent('at-risk-total', formatNumber(total));
        setTextContent('at-risk-critical', formatNumber(criticalCount));
        setTextContent('at-risk-avg-churn', avgChurn.toFixed(1) + '%');
        setTextContent('at-risk-total-ltv', formatCurrency(totalLtv));
    }

    function getActionLabel(action) {
        var normalized = (action || '').toUpperCase().replace(/[\s-]/g, '_');
        if (normalized.indexOf('BOOK') >= 0) return 'Book';
        if (normalized.indexOf('OFFER') >= 0 || normalized.indexOf('PROMOTION') >= 0) return 'Offer';
        if (normalized.indexOf('DISCOUNT') >= 0) return 'Discount';
        return 'Follow-up';
    }

    function getActionBtnClass(action) {
        var normalized = (action || '').toUpperCase().replace(/[\s-]/g, '_');
        if (normalized.indexOf('BOOK') >= 0) return 'btn-success';
        if (normalized.indexOf('OFFER') >= 0 || normalized.indexOf('PROMOTION') >= 0) return 'btn-offer';
        if (normalized.indexOf('DISCOUNT') >= 0) return 'btn-offer';
        return 'btn-follow-up';
    }

    /* ================================================================
       Customer Detail
       ================================================================ */

    function loadCustomerDetail(customerId) {
        if (!state.selectedSalonId) return;

        navigateTo('customer-detail');
        showLoading('view-customer-detail');

        fetchJSON('/v1/predictions/churn/' + customerId + '?salon_id=' + state.selectedSalonId)
            .then(function (data) {
                hideLoading('view-customer-detail');
                renderCustomerDetail(data);
            })
            .catch(function (err) {
                hideLoading('view-customer-detail');
                showError('view-customer-detail', 'Failed to load customer details.');
                console.error('Customer detail load error:', err);
            });
    }

    function renderCustomerDetail(data) {
        setTextContent('customer-name', data.customer_name || '--');
        setTextContent('customer-id', 'ID: ' + (data.customer_id || '--'));

        var riskBadge = document.getElementById('customer-risk-badge');
        var riskLevel = data.risk_level || 'LOW';
        riskBadge.textContent = riskLevel;
        riskBadge.className = 'badge badge-lg ' + getBadgeClass(riskLevel);

        var churnProb = data.churn_probability_90d || 0;
        renderChurnRing(churnProb, riskLevel);

        setTextContent('metric-next-visit', Math.round(data.predicted_next_visit_days || 0));
        setTextContent('metric-ltv', formatCurrency(data.estimated_ltv_12m || 0));
        setTextContent('metric-upsell', formatPercent(data.upsell_propensity || 0));

        renderRiskFactors(data.risk_factors || []);
        renderVisitHistory(data.visit_history || []);
    }

    function renderChurnRing(probability, riskLevel) {
        var ring = document.getElementById('churn-ring');
        var valueEl = document.getElementById('churn-ring-value');
        var pct = Math.min(100, Math.max(0, probability * 100));
        var color = getRiskColor(riskLevel);
        var deg = (pct / 100) * 360;

        ring.style.background = 'conic-gradient(' + color + ' 0deg, ' + color + ' ' + deg + 'deg, #E2E8F0 ' + deg + 'deg, #E2E8F0 360deg)';
        valueEl.textContent = pct.toFixed(1) + '%';
    }

    function renderRiskFactors(factors) {
        var container = document.getElementById('risk-factors-list');

        if (factors.length === 0) {
            container.innerHTML = '<p class="empty-state">No risk factors available.</p>';
            return;
        }

        container.innerHTML = factors.map(function (f, idx) {
            var impact = Math.min(100, Math.round((f.impact || 0) * 100));
            var severityClass = getSeverityClass(impact);
            return '<div class="risk-factor-item">' +
                '<span class="risk-factor-description">' + (idx + 1) + '. ' + (f.description || f.feature || '--') + '</span>' +
                '<div class="impact-bar-track"><div class="impact-bar ' + severityClass + '" style="width: ' + impact + '%"></div></div>' +
                '</div>';
        }).join('');
    }

    function renderVisitHistory(visits) {
        var tbody = document.getElementById('visit-history-tbody');

        if (visits.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No visit history available.</td></tr>';
            return;
        }

        tbody.innerHTML = visits.map(function (v) {
            var services = Array.isArray(v.services) ? v.services.join(', ') : (v.services || '--');
            return '<tr>' +
                '<td>' + formatDate(v.visit_date || '') + '</td>' +
                '<td>' + services + '</td>' +
                '<td>' + formatCurrency(v.amount || 0) + '</td>' +
                '<td>' + (v.stylist || '--') + '</td>' +
                '</tr>';
        }).join('');
    }

    /* ================================================================
       Analytics
       ================================================================ */

    function loadAnalytics() {
        if (!state.selectedSalonId) return;

        var sid = state.selectedSalonId;

        showLoading('view-analytics');

        Promise.all([
            fetchJSON('/v1/analytics/revenue/' + sid),
            fetchJSON('/v1/analytics/segments/' + sid),
            fetchJSON('/v1/analytics/services/' + sid)
        ])
            .then(function (results) {
                hideLoading('view-analytics');
                var revenue = results[0];
                var segments = results[1];
                var services = results[2];

                renderAnalyticsKPIs(revenue, segments);
                renderRevenueTrendChart(revenue);
                renderSegmentsChart(segments);
                renderServicesChart(services);
                renderVisitFrequencyChart(revenue);
                renderAnalyticsObservations(revenue, segments, services);
            })
            .catch(function (err) {
                hideLoading('view-analytics');
                showError('view-analytics', 'Failed to load analytics data.');
                console.error('Analytics load error:', err);
            });
    }

    function renderAnalyticsKPIs(revenue, segments) {
        var totals = revenue.totals || {};
        setTextContent('analytics-revenue', formatCurrency(totals.total_revenue || 0));
        setTextContent('analytics-visits', formatNumber(totals.total_visits || 0));
        setTextContent('analytics-ticket', formatCurrency(totals.avg_ticket || 0));
        setTextContent('analytics-customers', formatNumber(segments.total_customers || 0));
    }

    function renderRevenueTrendChart(data) {
        var months = data.months || [];
        var labels = months.map(function (m) { return m.month; });
        var values = months.map(function (m) { return m.revenue; });

        renderChart('chart-revenue-trend', {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Revenue',
                    data: values,
                    borderColor: '#D4A853',
                    backgroundColor: 'rgba(212, 168, 83, 0.12)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: '#D4A853',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return formatCurrencyShort(value);
                            }
                        }
                    }
                }
            }
        });
    }

    function renderSegmentsChart(data) {
        var segments = data.segments || [];
        var labels = segments.map(function (s) { return s.name; });
        var values = segments.map(function (s) { return s.count; });
        var colors = segments.map(function (s) { return s.color || '#D4A853'; });

        renderChart('chart-segments', {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: { family: 'Inter', size: 12 }
                        }
                    }
                }
            }
        });
    }

    function renderServicesChart(data) {
        var categories = data.categories || [];
        var labels = categories.map(function (c) { return c.category; });
        var values = categories.map(function (c) { return c.revenue; });

        renderChart('chart-services', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Revenue',
                    data: values,
                    backgroundColor: '#D4A853',
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return formatCurrencyShort(value);
                            }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    function renderVisitFrequencyChart(data) {
        var months = data.months || [];
        var labels = months.map(function (m) { return m.month; });
        var values = months.map(function (m) { return m.visit_count; });

        renderChart('chart-visit-freq', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Visits',
                    data: values,
                    backgroundColor: 'rgba(212, 168, 83, 0.7)',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });
    }

    function renderAnalyticsObservations(revenue, segments, services) {
        var container = document.getElementById('analytics-observations-list');
        if (!container) return;

        var observations = [];
        var totals = revenue.totals || {};
        var months = revenue.months || [];
        var segmentList = segments.segments || [];
        var categoryList = services.categories || [];

        if (months.length > 0 && totals.total_revenue) {
            var avgRevenue = Math.round(totals.total_revenue / months.length);
            observations.push(
                'Revenue averaged ' + formatCurrency(avgRevenue) + ' per month over the past ' + months.length + ' months.'
            );
        }

        if (segmentList.length > 0) {
            var totalCust = segments.total_customers || 0;
            var topSegment = segmentList[0];
            for (var i = 1; i < segmentList.length; i++) {
                if (segmentList[i].count > topSegment.count) {
                    topSegment = segmentList[i];
                }
            }
            if (totalCust > 0) {
                var segPct = ((topSegment.count / totalCust) * 100).toFixed(1);
                observations.push(
                    topSegment.name + ' is the largest customer segment at ' + segPct + '%.'
                );
            }
        }

        if (categoryList.length > 0) {
            var totalServiceRev = categoryList.reduce(function (acc, c) { return acc + (c.revenue || 0); }, 0);
            var topService = categoryList[0];
            for (var j = 1; j < categoryList.length; j++) {
                if (categoryList[j].revenue > topService.revenue) {
                    topService = categoryList[j];
                }
            }
            if (totalServiceRev > 0) {
                var svcPct = ((topService.revenue / totalServiceRev) * 100).toFixed(1);
                observations.push(
                    topService.category + ' category generates ' + svcPct + '% of all service revenue.'
                );
            }
        }

        if (totals.total_visits && totals.avg_ticket) {
            observations.push(
                formatNumber(totals.total_visits) + ' total visits recorded with an average ticket of ' + formatCurrency(totals.avg_ticket) + '.'
            );
        }

        if (observations.length === 0) {
            container.innerHTML = '<div class="insight-item">Insufficient data to generate observations.</div>';
            return;
        }

        container.innerHTML = observations.map(function (text) {
            return '<div class="insight-item">' + text + '</div>';
        }).join('');
    }

    /* ================================================================
       Stylist Performance
       ================================================================ */

    function loadStylists() {
        if (!state.selectedSalonId) return;

        showLoading('view-stylists');

        fetchJSON('/v1/stylists/' + state.selectedSalonId)
            .then(function (data) {
                hideLoading('view-stylists');
                renderStylistSummary(data.summary || {});
                renderStylistsTable(data.stylists || []);
                renderStylistRevenueChart(data.stylists || []);
            })
            .catch(function (err) {
                hideLoading('view-stylists');
                showError('view-stylists', 'Failed to load stylist data.');
                console.error('Stylists load error:', err);
            });
    }

    function renderStylistSummary(summary) {
        setTextContent('stylist-count', summary.total_stylists || 0);
        setTextContent('stylist-avg-revenue', formatCurrency(summary.avg_revenue_per_stylist || 0));
        setTextContent('stylist-top', summary.top_performer || '--');
    }

    function renderStylistsTable(stylists) {
        var tbody = document.getElementById('stylists-tbody');

        if (stylists.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No stylist data available.</td></tr>';
            return;
        }

        tbody.innerHTML = stylists.map(function (s, idx) {
            var retentionClass = '';
            var retention = s.retention_rate || 0;
            if (retention >= 80) {
                retentionClass = 'text-success';
            } else if (retention >= 60) {
                retentionClass = 'text-warning';
            } else {
                retentionClass = 'text-danger';
            }

            var rankHtml = '';
            if (idx < 3) {
                rankHtml = '<span class="rank-badge rank-' + (idx + 1) + '">' + (idx + 1) + '</span>';
            }

            return '<tr>' +
                '<td>' + rankHtml + (s.stylist_name || '--') + '</td>' +
                '<td>' + (s.specialty || '--') + '</td>' +
                '<td>' + (s.experience_years || 0) + ' yrs</td>' +
                '<td>' + formatNumber(s.customer_count || 0) + '</td>' +
                '<td>' + formatNumber(s.visit_count || 0) + '</td>' +
                '<td>' + formatCurrency(s.total_revenue || 0) + '</td>' +
                '<td>' + formatCurrency(s.avg_ticket || 0) + '</td>' +
                '<td class="' + retentionClass + '">' + formatPercent(retention) + '</td>' +
                '</tr>';
        }).join('');
    }

    function renderStylistRevenueChart(stylists) {
        var labels = stylists.map(function (s) { return s.stylist_name; });
        var values = stylists.map(function (s) { return s.total_revenue; });

        renderChart('chart-stylist-revenue', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Revenue',
                    data: values,
                    backgroundColor: '#D4A853',
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return formatCurrencyShort(value);
                            }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    /* ================================================================
       System Monitoring
       ================================================================ */

    function loadMonitoring() {
        showLoading('view-monitoring');

        fetchJSON('/v1/system/status')
            .then(function (data) {
                hideLoading('view-monitoring');
                renderSystemStatus(data.services || {});
                renderModelsTable(data.models || {});
                renderPredictionDistChart(data.prediction_distribution || {});
                renderDataOverview(data.data || {});
            })
            .catch(function (err) {
                hideLoading('view-monitoring');
                showError('view-monitoring', 'Failed to load system status.');
                console.error('Monitoring load error:', err);
            });
    }

    function renderSystemStatus(services) {
        renderStatusCard('status-db', services.database || {}, 'Database');
        renderStatusCard('status-redis', services.redis || {}, 'Redis Cache');
        renderStatusCard('status-features', services.feature_store || {}, 'Feature Store');
        renderStatusCard('status-predictions', services.prediction_engine || {}, 'Prediction Engine');
    }

    function renderStatusCard(elementId, service, name) {
        var card = document.getElementById(elementId);
        if (!card) return;

        var dot = card.querySelector('.status-dot');
        var detail = card.querySelector('.status-detail');

        var isUp = service.status === 'connected' || service.status === 'operational';
        dot.className = 'status-dot ' + (isUp ? 'connected' : 'error');

        var detailText = service.status || 'unknown';
        if (service.latency_ms !== undefined) {
            detailText += ' | ' + service.latency_ms + 'ms latency';
        }
        if (service.keys !== undefined) {
            detailText += ' | ' + formatNumber(service.keys) + ' keys';
        }
        if (service.features_computed !== undefined) {
            detailText += ' | ' + formatNumber(service.features_computed) + ' features';
        }
        if (service.predictions_stored !== undefined) {
            detailText += ' | ' + formatNumber(service.predictions_stored) + ' predictions';
        }
        if (service.last_updated) {
            detailText += ' | updated ' + formatTimestamp(service.last_updated);
        }
        if (service.last_run) {
            detailText += ' | last run ' + formatTimestamp(service.last_run);
        }

        detail.textContent = detailText;
    }

    function renderModelsTable(models) {
        var tbody = document.getElementById('models-tbody');
        var modelEntries = Object.keys(models);

        if (modelEntries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No model data available.</td></tr>';
            return;
        }

        tbody.innerHTML = modelEntries.map(function (key) {
            var m = models[key];
            var metrics = m.metrics || {};
            var primaryMetric = Object.keys(metrics)[0] || '--';
            var primaryValue = metrics[primaryMetric] !== undefined ? metrics[primaryMetric] : '--';

            return '<tr>' +
                '<td>' + key + '</td>' +
                '<td>' + (m.type || '--') + '</td>' +
                '<td>' + (m.version || '--') + '</td>' +
                '<td>' + primaryMetric + '</td>' +
                '<td>' + (typeof primaryValue === 'number' ? primaryValue.toFixed(4) : primaryValue) + '</td>' +
                '<td>' + formatTimestamp(m.trained_at || '') + '</td>' +
                '</tr>';
        }).join('');
    }

    function renderPredictionDistChart(dist) {
        var labels = ['Low', 'Medium', 'High', 'Critical'];
        var values = [
            dist.LOW || 0,
            dist.MEDIUM || 0,
            dist.HIGH || 0,
            dist.CRITICAL || 0
        ];

        renderChart('chart-pred-dist', {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        RISK_COLORS.LOW,
                        RISK_COLORS.MEDIUM,
                        RISK_COLORS.HIGH,
                        RISK_COLORS.CRITICAL
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: { family: 'Inter', size: 12 }
                        }
                    }
                }
            }
        });
    }

    function renderDataOverview(data) {
        setTextContent('data-customers', formatNumber(data.total_customers || 0));
        setTextContent('data-visits', formatNumber(data.total_visits || 0));
        setTextContent('data-salons', formatNumber(data.total_salons || 0));
    }

    /* ================================================================
       Case Study
       ================================================================ */

    function loadCaseStudy() {
        showLoading('view-case-study');

        fetchJSON('/v1/salon/SALON_001/case-study')
            .then(function (data) {
                hideLoading('view-case-study');
                renderCaseStudy(data);
            })
            .catch(function (err) {
                hideLoading('view-case-study');
                console.warn('Case study API unavailable, rendering fallback data:', err);
                renderCaseStudyFallback();
            });
    }

    function renderCaseStudy(data) {
        var metrics = buildCaseStudyMetrics(data);
        renderComparisonGrid(metrics);
        renderCaseStudyChart(metrics);
        renderKeyInsights(data);
        renderNetworkProjection(data);
    }

    function buildCaseStudyMetrics(data) {
        if (data.improvements && data.improvements.length > 0) {
            return data.improvements.map(function (imp) {
                return {
                    key: imp.metric,
                    name: METRIC_NAMES[imp.metric] || imp.metric,
                    before: imp.before,
                    after: imp.after,
                    unit: METRIC_UNITS[imp.metric] || imp.unit || '',
                    lower_is_better: LOWER_IS_BETTER[imp.metric] || false
                };
            });
        }

        if (data.before && data.after) {
            var keys = Object.keys(data.before);
            return keys.map(function (key) {
                return {
                    key: key,
                    name: METRIC_NAMES[key] || key,
                    before: data.before[key],
                    after: data.after[key],
                    unit: METRIC_UNITS[key] || '',
                    lower_is_better: LOWER_IS_BETTER[key] || false
                };
            });
        }

        return [];
    }

    function renderCaseStudyFallback() {
        var metrics = [
            { key: 'churn_rate', name: 'Customer Churn Rate', before: 18.5, after: 11.2, unit: 'percent', lower_is_better: true },
            { key: 'avg_inter_visit_days', name: 'Avg Visit Interval', before: 42.0, after: 36.0, unit: 'days', lower_is_better: true },
            { key: 'reengagement_rate', name: 'Re-engagement Rate', before: 12.0, after: 38.0, unit: 'percent', lower_is_better: false },
            { key: 'revenue_per_customer_year', name: 'Revenue per Customer/Year', before: 52000, after: 61500, unit: 'jpy', lower_is_better: false },
            { key: 'admin_time_per_day', name: 'Admin Time per Day', before: 25.0, after: 10.0, unit: 'minutes', lower_is_better: true },
            { key: 'customer_retention_12m', name: '12-Month Customer Retention', before: 68.0, after: 82.0, unit: 'percent', lower_is_better: false },
            { key: 'at_risk_identification_rate', name: 'At-Risk Identification Rate', before: 15.0, after: 94.0, unit: 'percent', lower_is_better: false },
            { key: 'campaign_response_rate', name: 'Campaign Response Rate', before: 4.2, after: 11.8, unit: 'percent', lower_is_better: false },
            { key: 'avg_ticket_value', name: 'Average Ticket Value', before: 8500, after: 9800, unit: 'jpy', lower_is_better: false },
            { key: 'noshow_rate', name: 'No-Show Rate', before: 12.0, after: 7.2, unit: 'percent', lower_is_better: true },
            { key: 'active_customers', name: 'Active Customers', before: 285, after: 342, unit: 'count', lower_is_better: false },
            { key: 'monthly_revenue', name: 'Monthly Revenue', before: 2420000, after: 3150000, unit: 'jpy', lower_is_better: false }
        ];

        renderComparisonGrid(metrics);
        renderCaseStudyChart(metrics);
        renderKeyInsightsFallback();
        renderNetworkProjectionFromMetrics(metrics);
    }

    function renderComparisonGrid(metrics) {
        var grid = document.getElementById('comparison-grid');

        if (metrics.length === 0) {
            grid.innerHTML = '<p class="empty-state">No comparison data available.</p>';
            return;
        }

        grid.innerHTML = metrics.map(function (m) {
            var changePct = ((m.after - m.before) / Math.abs(m.before)) * 100;
            var isImprovement = m.lower_is_better ? changePct < 0 : changePct > 0;
            var badgeClass = isImprovement ? 'positive' : 'negative';
            var sign = changePct > 0 ? '+' : '';

            return '<div class="comparison-row">' +
                '<span class="comparison-metric">' + m.name + '</span>' +
                '<span class="comparison-before">' + formatMetricValue(m.before, m.unit) + '</span>' +
                '<span class="comparison-arrow">&#8594;</span>' +
                '<span class="comparison-after">' + formatMetricValue(m.after, m.unit) + '</span>' +
                '<span class="change-badge ' + badgeClass + '">' + sign + changePct.toFixed(1) + '%</span>' +
                '</div>';
        }).join('');
    }

    function renderCaseStudyChart(metrics) {
        var chartMetrics = metrics.filter(function (m) {
            return m.unit === 'percent';
        }).slice(0, 6);

        var labels = chartMetrics.map(function (m) { return m.name; });
        var beforeData = chartMetrics.map(function (m) { return m.before; });
        var afterData = chartMetrics.map(function (m) { return m.after; });

        renderChart('chart-case-study', {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Before',
                        data: beforeData,
                        backgroundColor: '#94A3B8',
                        borderRadius: 6
                    },
                    {
                        label: 'After',
                        data: afterData,
                        backgroundColor: '#D4A853',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: { family: 'Inter', size: 12 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            maxRotation: 30
                        }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });
    }

    function renderKeyInsights(data) {
        var list = document.getElementById('key-insights-list');
        var insights = [];

        if (data.before && data.after) {
            var churnReduction = ((data.before.churn_rate - data.after.churn_rate) / data.before.churn_rate * 100).toFixed(1);
            var idRate = (data.after.at_risk_identification_rate || 0).toFixed(0);
            var adminSaved = (data.before.admin_time_per_day - data.after.admin_time_per_day).toFixed(0);
            var revUplift = ((data.after.monthly_revenue - data.before.monthly_revenue) / data.before.monthly_revenue * 100).toFixed(1);
            var campaignBefore = (data.before.campaign_response_rate || 0).toFixed(1);
            var campaignAfter = (data.after.campaign_response_rate || 0).toFixed(1);
            var noshowReduction = ((data.before.noshow_rate - data.after.noshow_rate) / data.before.noshow_rate * 100).toFixed(1);

            insights = [
                'At-risk identification rate improved to ' + idRate + '%, directly enabling a ' + churnReduction + '% churn reduction through timely stylist intervention.',
                'Automated insights reduced administrative overhead by ' + adminSaved + ' minutes per day per stylist, freeing time for direct client relationship building.',
                'Revenue uplift of ' + revUplift + '% monthly demonstrates that retention-driven growth is significantly more capital-efficient than acquisition-driven strategies.',
                'Campaign response rates improved from ' + campaignBefore + '% to ' + campaignAfter + '% due to ML-personalized timing and offer selection.',
                'No-show rates decreased by ' + noshowReduction + '% as predictive scheduling enabled proactive confirmation outreach.',
                'The compounding effect of higher retention and higher per-customer revenue generated substantial monthly revenue uplift.'
            ];
        }

        if (insights.length === 0) {
            list.innerHTML = '<li class="empty-state">No insights available.</li>';
            return;
        }

        list.innerHTML = insights.map(function (insight) {
            return '<li>' + insight + '</li>';
        }).join('');
    }

    function renderKeyInsightsFallback() {
        var list = document.getElementById('key-insights-list');
        var insights = [
            'Early identification is the primary churn reduction lever -- the improvement in at-risk identification rate directly enabled the 39.5% churn reduction through timely stylist intervention.',
            'Automated insights reduced administrative overhead by 60%, freeing an average of 15 minutes per day per stylist for direct client relationship building.',
            'Revenue uplift of 30.2% monthly demonstrates that retention-driven growth is significantly more capital-efficient than acquisition-driven growth strategies.',
            'Campaign response rates improved significantly due to ML-personalized timing and offer selection, validating the contextual recommendation engine.',
            'No-show rates decreased as predictive scheduling enabled proactive confirmation outreach to high-risk appointments.',
            'The compounding effect of higher retention and higher per-customer revenue generated a monthly revenue increase of approximately 730,000 JPY.'
        ];

        list.innerHTML = insights.map(function (insight) {
            return '<li>' + insight + '</li>';
        }).join('');
    }

    function renderNetworkProjection(data) {
        if (!data.before || !data.after) {
            renderNetworkProjectionFromMetrics([]);
            return;
        }

        var monthlyUpliftPerSalon = data.after.monthly_revenue - data.before.monthly_revenue;
        var annualNetworkUplift = monthlyUpliftPerSalon * 12 * 10000;
        var retainedPerSalon = Math.round(data.before.active_customers * (data.before.churn_rate - data.after.churn_rate) / 100);
        var retainedNetwork = retainedPerSalon * 10000;
        var churnReductionPct = ((data.before.churn_rate - data.after.churn_rate) / data.before.churn_rate * 100);

        var container = document.getElementById('network-projection');
        container.innerHTML =
            buildProjectionItem('+' + formatLargeJPY(annualNetworkUplift), 'Aggregate Revenue Uplift (Annual)') +
            buildProjectionItem('~' + NUMBER_FORMATTER.format(retainedNetwork), 'Customers Retained Annually') +
            buildProjectionItem(churnReductionPct.toFixed(1) + '%', 'Avg Churn Reduction');
    }

    function renderNetworkProjectionFromMetrics(metrics) {
        var monthlyRevBefore = 2420000;
        var monthlyRevAfter = 3150000;
        var churnBefore = 18.5;
        var churnAfter = 11.2;
        var activeBefore = 285;

        metrics.forEach(function (m) {
            if (m.key === 'monthly_revenue') {
                monthlyRevBefore = m.before;
                monthlyRevAfter = m.after;
            }
            if (m.key === 'churn_rate') {
                churnBefore = m.before;
                churnAfter = m.after;
            }
            if (m.key === 'active_customers') {
                activeBefore = m.before;
            }
        });

        var monthlyUpliftPerSalon = monthlyRevAfter - monthlyRevBefore;
        var annualNetworkUplift = monthlyUpliftPerSalon * 12 * 10000;
        var retainedPerSalon = Math.round(activeBefore * (churnBefore - churnAfter) / 100);
        var retainedNetwork = retainedPerSalon * 10000;
        var churnReductionPct = ((churnBefore - churnAfter) / churnBefore * 100);

        var container = document.getElementById('network-projection');
        container.innerHTML =
            buildProjectionItem('+' + formatLargeJPY(annualNetworkUplift), 'Aggregate Revenue Uplift (Annual)') +
            buildProjectionItem('~' + NUMBER_FORMATTER.format(retainedNetwork), 'Customers Retained Annually') +
            buildProjectionItem(churnReductionPct.toFixed(1) + '%', 'Avg Churn Reduction');
    }

    function buildProjectionItem(value, label) {
        return '<div class="projection-item">' +
            '<div class="projection-value">' + value + '</div>' +
            '<div class="projection-label">' + label + '</div>' +
            '</div>';
    }

    /* ================================================================
       Chart Utilities
       ================================================================ */

    function renderChart(canvasId, config) {
        if (state.charts[canvasId]) {
            state.charts[canvasId].destroy();
            state.charts[canvasId] = null;
        }

        var canvas = document.getElementById(canvasId);
        if (!canvas) return;

        var ctx = canvas.getContext('2d');

        Chart.defaults.font.family = 'Inter';

        state.charts[canvasId] = new Chart(ctx, config);
    }

    function destroyAllCharts() {
        Object.keys(state.charts).forEach(function (key) {
            if (state.charts[key]) {
                state.charts[key].destroy();
                state.charts[key] = null;
            }
        });
    }

    /* ================================================================
       Formatters
       ================================================================ */

    function formatCurrency(value) {
        return JPY_FORMATTER.format(value || 0);
    }

    function formatCurrencyShort(value) {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        }
        if (value >= 1000) {
            return (value / 1000).toFixed(0) + 'K';
        }
        return String(value);
    }

    function formatPercent(value) {
        var pct = (typeof value === 'number' && value >= 0 && value <= 1)
            ? value * 100
            : value;
        return Number(pct).toFixed(1) + '%';
    }

    function formatNumber(value) {
        return NUMBER_FORMATTER.format(value || 0);
    }

    function formatDate(str) {
        if (!str) return '--';
        try {
            var d = new Date(str);
            if (isNaN(d.getTime())) return str;
            return d.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (e) {
            return str;
        }
    }

    function formatTimestamp(str) {
        if (!str) return '--';
        try {
            var d = new Date(str);
            if (isNaN(d.getTime())) return str;
            var formatted = d.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            }) + ' ' + d.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
            var now = new Date();
            var diffMs = now.getTime() - d.getTime();
            var diffMins = Math.floor(diffMs / 60000);
            var relative = '';
            if (diffMins < 1) {
                relative = 'just now';
            } else if (diffMins < 60) {
                relative = diffMins + ' min ago';
            } else if (diffMins < 1440) {
                var hours = Math.floor(diffMins / 60);
                relative = hours + ' hour' + (hours !== 1 ? 's' : '') + ' ago';
            } else {
                var days = Math.floor(diffMins / 1440);
                relative = days + ' day' + (days !== 1 ? 's' : '') + ' ago';
            }
            return formatted + ' (' + relative + ')';
        } catch (e) {
            return str;
        }
    }

    function formatMetricValue(value, unit) {
        switch (unit) {
            case 'jpy':
                return JPY_FORMATTER.format(value);
            case 'percent':
                return value.toFixed(1) + '%';
            case 'minutes':
                return value.toFixed(0) + ' min';
            case 'days':
                return value.toFixed(1) + ' days';
            case 'count':
                return NUMBER_FORMATTER.format(value);
            default:
                return String(value);
        }
    }

    function formatLargeJPY(value) {
        var billions = value / 1000000000;
        if (billions >= 1) {
            return billions.toFixed(1) + 'B JPY';
        }
        var millions = value / 1000000;
        return millions.toFixed(0) + 'M JPY';
    }

    function getRiskColor(level) {
        return RISK_COLORS[(level || '').toUpperCase()] || '#94A3B8';
    }

    function getBadgeClass(riskLevel) {
        var level = (riskLevel || 'low').toLowerCase();
        return 'badge-' + level;
    }

    function getSeverityClass(impact) {
        if (impact >= 75) return 'severity-critical';
        if (impact >= 50) return 'severity-high';
        if (impact >= 25) return 'severity-medium';
        return 'severity-low';
    }

    /* ================================================================
       DOM Utilities
       ================================================================ */

    function setTextContent(elementId, text) {
        var el = document.getElementById(elementId);
        if (el) {
            el.textContent = text;
        }
    }

    /* ================================================================
       Loading & Error States
       ================================================================ */

    function showLoading(containerId) {
        var container = document.getElementById(containerId);
        if (!container) return;
        if (container.querySelector('.loading')) return;

        var spinner = document.createElement('div');
        spinner.className = 'loading';
        container.insertBefore(spinner, container.firstChild);
    }

    function hideLoading(containerId) {
        var container = document.getElementById(containerId);
        if (!container) return;

        var spinner = container.querySelector('.loading');
        if (spinner) spinner.remove();
    }

    function showError(containerId, message) {
        var container = document.getElementById(containerId);
        if (!container) return;

        var existing = container.querySelector('.error-state');
        if (!existing) {
            var errorDiv = document.createElement('div');
            errorDiv.className = 'error-state';
            errorDiv.textContent = message;
            container.insertBefore(errorDiv, container.firstChild);
        } else {
            existing.textContent = message;
        }
    }

    /* ================================================================
       Bootstrap
       ================================================================ */

    document.addEventListener('DOMContentLoaded', function () {
        setupLoginForm();
    });

})();
