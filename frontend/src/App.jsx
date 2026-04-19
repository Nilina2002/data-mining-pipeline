import { useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const SEGMENTS = {
    "Loyal Customers": {
        color: "#00C896",
        bg: "rgba(0,200,150,0.08)",
        border: "rgba(0,200,150,0.3)",
        badge: "HIGH VALUE",
        risk: "Low churn risk",
        description:
            "Recently active, frequent customers with high spend. Protect this group with loyalty rewards and targeted upsell.",
        actions: [
            "Enrol in a VIP loyalty programme",
            "Offer early access to new product launches",
            "Send personalised product recommendations",
            "Use premium retention offers sparingly",
        ],
    },
    "Potential Loyalists": {
        color: "#3B9EFF",
        bg: "rgba(59,158,255,0.08)",
        border: "rgba(59,158,255,0.3)",
        badge: "GROWTH OPPORTUNITY",
        risk: "Medium churn risk",
        description:
            "Customers showing healthy activity but not yet fully loyal. Small nudges can move this segment upward quickly.",
        actions: [
            "Send a first-repeat-purchase offer",
            "Recommend complementary products",
            "Invite them into a feedback programme",
            "Follow up with a timed re-engagement email",
        ],
    },
    "Lost / Inactive": {
        color: "#FF5C5C",
        bg: "rgba(255,92,92,0.08)",
        border: "rgba(255,92,92,0.3)",
        badge: "WIN-BACK REQUIRED",
        risk: "High churn risk",
        description:
            "Customers who have not purchased recently and are unlikely to return without a focused win-back campaign.",
        actions: [
            "Trigger a reactivation campaign",
            "Offer a meaningful return incentive",
            "Survey for churn reasons",
            "Compare win-back cost against acquisition cost",
        ],
    },
};

const DEFAULT_STATS = {
    total_customers: 4338,
    recency_min: 1,
    recency_max: 369,
    frequency_min: 1,
    frequency_max: 30,
    monetary_min: 3.75,
    monetary_max: 19880.99570000001,
    classes: ["Lost / Inactive", "Loyal Customers", "Potential Loyalists"],
    segment_counts: {
        "Potential Loyalists": 1876,
        "Loyal Customers": 1481,
        "Lost / Inactive": 981,
    },
};

function AnimatedNumber({ value, prefix = "", suffix = "", decimals = 0 }) {
    const [display, setDisplay] = useState(0);

    useEffect(() => {
        let frame = 0;
        const start = performance.now();

        const step = (now) => {
            const progress = Math.min((now - start) / 800, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplay(value * eased);
            if (progress < 1) frame = requestAnimationFrame(step);
        };

        frame = requestAnimationFrame(step);
        return () => cancelAnimationFrame(frame);
    }, [value]);

    const formatted =
        decimals === 0 ? Math.round(display).toLocaleString() : display.toFixed(decimals);

    return (
        <span>
            {prefix}
            {formatted}
            {suffix}
        </span>
    );
}

function ConfidenceBar({ label, value, color, isTop }) {
    const [width, setWidth] = useState(0);

    useEffect(() => {
        const timer = window.setTimeout(() => setWidth(value * 100), 100);
        return () => window.clearTimeout(timer);
    }, [value]);

    return (
        <div style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span
                    style={{
                        fontSize: 12,
                        color: isTop ? "#fff" : "#9aa0aa",
                        fontFamily: "DM Mono, monospace",
                        fontWeight: isTop ? 700 : 400,
                    }}
                >
                    {label}
                </span>
                <span
                    style={{
                        fontSize: 12,
                        color: isTop ? color : "#7f8794",
                        fontFamily: "DM Mono, monospace",
                        fontWeight: 700,
                    }}
                >
                    {(value * 100).toFixed(1)}%
                </span>
            </div>
            <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
                <div
                    style={{
                        height: "100%",
                        width: `${width}%`,
                        background: isTop ? color : "rgba(255,255,255,0.15)",
                        borderRadius: 3,
                        transition: "width 0.8s cubic-bezier(0.34,1.56,0.64,1)",
                        boxShadow: isTop ? `0 0 12px ${color}80` : "none",
                    }}
                />
            </div>
        </div>
    );
}

function RFMSlider({ label, sublabel, value, min, max, step, onChange, unit, color, icon, avgValue }) {
    const pct = ((value - min) / (max - min)) * 100;
    const avgPct = ((avgValue - min) / (max - min)) * 100;

    return (
        <div
            style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 14,
                padding: "18px 20px",
                marginBottom: 12,
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                        <span style={{ fontSize: 18 }}>{icon}</span>
                        <span
                            style={{
                                fontSize: 13,
                                fontWeight: 700,
                                color: "#e0e0e0",
                                letterSpacing: "0.05em",
                                textTransform: "uppercase",
                                fontFamily: "DM Mono, monospace",
                            }}
                        >
                            {label}
                        </span>
                    </div>
                    <div style={{ fontSize: 11, color: "#7f8794", fontFamily: "DM Mono, monospace", paddingLeft: 26 }}>
                        {sublabel}
                    </div>
                </div>
                <div
                    style={{
                        background: `${color}18`,
                        border: `1px solid ${color}40`,
                        borderRadius: 8,
                        padding: "6px 14px",
                        textAlign: "center",
                        minWidth: 80,
                    }}
                >
                    <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "DM Mono, monospace", lineHeight: 1 }}>
                        {unit === "£" ? `£${value.toLocaleString()}` : value.toLocaleString()}
                        {unit && unit !== "£" ? ` ${unit}` : ""}
                    </div>
                </div>
            </div>

            <div style={{ position: "relative", paddingBottom: 18 }}>
                <div style={{ position: "relative", height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, marginBottom: 4 }}>
                    <div
                        style={{
                            position: "absolute",
                            left: 0,
                            width: `${pct}%`,
                            height: "100%",
                            background: `linear-gradient(90deg, ${color}60, ${color})`,
                            borderRadius: 3,
                            transition: "width 0.1s",
                        }}
                    />
                    <div
                        style={{
                            position: "absolute",
                            left: `${avgPct}%`,
                            top: -3,
                            width: 2,
                            height: 12,
                            background: "rgba(255,255,255,0.3)",
                            borderRadius: 1,
                            transform: "translateX(-50%)",
                        }}
                    />
                </div>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(event) => onChange(Number(event.target.value))}
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: 6, opacity: 0, cursor: "pointer", margin: 0 }}
                />
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 10, color: "#4d5360", fontFamily: "DM Mono, monospace" }}>{min}</span>
                    <span style={{ fontSize: 10, color: "#4d5360", fontFamily: "DM Mono, monospace" }}>
                        avg: {unit === "£" ? `£${avgValue.toLocaleString()}` : `${avgValue} days`}
                    </span>
                    <span style={{ fontSize: 10, color: "#4d5360", fontFamily: "DM Mono, monospace" }}>{max}</span>
                </div>
            </div>
        </div>
    );
}

function getTopProbability(probabilities) {
    return Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1])[0] || [null, 0];
}

export default function CustomerPortal() {
    const [stats, setStats] = useState(DEFAULT_STATS);
    const [recency, setRecency] = useState(55);
    const [frequency, setFrequency] = useState(3);
    const [monetary, setMonetary] = useState(850);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");
    const [predicting, setPredicting] = useState(false);
    const [predicted, setPredicted] = useState(false);
    const [activeTab, setActiveTab] = useState("predict");

    useEffect(() => {
        let isMounted = true;

        async function loadStats() {
            try {
                const response = await fetch(`${API_BASE_URL}/model/stats`);
                if (!response.ok) {
                    throw new Error(`Stats request failed: ${response.status}`);
                }
                const data = await response.json();
                if (isMounted) {
                    setStats(data);
                    setRecency(Math.min(55, data.recency_max));
                    setFrequency(Math.min(3, data.frequency_max));
                    setMonetary(Math.min(850, Math.round(data.monetary_max)));
                }
            } catch {
                if (isMounted) {
                    setError("Unable to load model stats. Start the FastAPI service first.");
                }
            }
        }

        loadStats();
        return () => {
            isMounted = false;
        };
    }, []);

    const handlePredict = async () => {
        setPredicting(true);
        setPredicted(false);
        setError("");

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ recency, frequency, monetary }),
            });

            if (!response.ok) {
                throw new Error(`Prediction request failed: ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
            setPredicted(true);
        } catch (requestError) {
            setError(requestError.message || "Prediction failed.");
        } finally {
            setPredicting(false);
        }
    };

    const seg = result ? SEGMENTS[result.predicted] : null;
    const quickProfiles = [
        { name: "Champion Buyer", r: 12, f: 18, m: 8500, tag: "VIP" },
        { name: "Casual Browser", r: 45, f: 2, m: 320, tag: "LOW" },
        { name: "Lapsed Customer", r: 280, f: 1, m: 210, tag: "WIN-BACK" },
        { name: "Rising Star", r: 22, f: 6, m: 1800, tag: "PROMISING" },
    ];

    const range = {
        recency: [stats.recency_min, stats.recency_max],
        frequency: [stats.frequency_min, stats.frequency_max],
        monetary: [stats.monetary_min, stats.monetary_max],
    };

    return (
        <div
            style={{
                minHeight: "100vh",
                background: "#090b12",
                color: "#e0e0e0",
                paddingBottom: 40,
            }}
        >
            <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { to{transform:rotate(360deg)} }
      `}</style>

            <div
                style={{
                    background: "rgba(255,255,255,0.02)",
                    borderBottom: "1px solid rgba(255,255,255,0.06)",
                    padding: "16px 32px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 16,
                    flexWrap: "wrap",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div
                        style={{
                            width: 36,
                            height: 36,
                            background: "linear-gradient(135deg, #00C896, #3B9EFF)",
                            borderRadius: 10,
                        }}
                    />
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.02em" }}>RFM Segment Predictor</div>
                        <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace" }}>
                            React frontend + FastAPI model service
                        </div>
                    </div>
                </div>

                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {["predict", "stats", "about"].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            style={{
                                background: activeTab === tab ? "rgba(255,255,255,0.08)" : "transparent",
                                border: `1px solid ${activeTab === tab ? "rgba(255,255,255,0.15)" : "transparent"}`,
                                color: activeTab === tab ? "#fff" : "#808694",
                                borderRadius: 8,
                                padding: "6px 14px",
                                cursor: "pointer",
                                fontSize: 12,
                                fontWeight: 600,
                                textTransform: "capitalize",
                            }}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </div>

            {activeTab === "predict" && (
                <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 24px" }}>
                    {error ? (
                        <div
                            style={{
                                background: "rgba(255,92,92,0.08)",
                                border: "1px solid rgba(255,92,92,0.22)",
                                color: "#ff8d8d",
                                padding: 14,
                                borderRadius: 12,
                                marginBottom: 20,
                            }}
                        >
                            {error}
                        </div>
                    ) : null}

                    <div style={{ marginBottom: 24 }}>
                        <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 10, letterSpacing: "0.08em" }}>
                            QUICK LOAD - TEST PROFILES
                        </div>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            {quickProfiles.map((profile) => (
                                <button
                                    key={profile.name}
                                    onClick={() => {
                                        setRecency(profile.r);
                                        setFrequency(profile.f);
                                        setMonetary(profile.m);
                                        setPredicted(false);
                                        setResult(null);
                                    }}
                                    style={{
                                        background: "rgba(255,255,255,0.04)",
                                        border: "1px solid rgba(255,255,255,0.08)",
                                        color: "#c8cdd6",
                                        borderRadius: 20,
                                        padding: "6px 14px",
                                        cursor: "pointer",
                                        fontSize: 12,
                                        fontWeight: 500,
                                    }}
                                >
                                    {profile.name} - {profile.tag}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                        <div>
                            <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 12, letterSpacing: "0.08em" }}>
                                RFM INPUT FEATURES
                            </div>

                            <RFMSlider
                                label="Recency"
                                sublabel="Days since last purchase (lower is better)"
                                icon="C"
                                value={recency}
                                min={range.recency[0]}
                                max={range.recency[1]}
                                step={1}
                                onChange={(nextValue) => {
                                    setRecency(nextValue);
                                    setPredicted(false);
                                    setResult(null);
                                }}
                                color="#FF5C5C"
                                avgValue={93}
                            />

                            <RFMSlider
                                label="Frequency"
                                sublabel="Number of unique orders placed"
                                icon="F"
                                value={frequency}
                                min={range.frequency[0]}
                                max={range.frequency[1]}
                                step={1}
                                onChange={(nextValue) => {
                                    setFrequency(nextValue);
                                    setPredicted(false);
                                    setResult(null);
                                }}
                                unit="orders"
                                color="#3B9EFF"
                                avgValue={4}
                            />

                            <RFMSlider
                                label="Monetary"
                                sublabel="Total spend in GBP"
                                icon="M"
                                value={monetary}
                                min={Math.floor(range.monetary[0])}
                                max={Math.ceil(range.monetary[1])}
                                step={50}
                                onChange={(nextValue) => {
                                    setMonetary(nextValue);
                                    setPredicted(false);
                                    setResult(null);
                                }}
                                unit="GBP"
                                color="#F59E0B"
                                avgValue={2054}
                            />

                            <button
                                onClick={handlePredict}
                                disabled={predicting}
                                style={{
                                    width: "100%",
                                    padding: "14px 0",
                                    marginTop: 4,
                                    background: predicting ? "rgba(0,200,150,0.1)" : "linear-gradient(135deg, #00C896, #00A878)",
                                    border: "1px solid rgba(0,200,150,0.4)",
                                    borderRadius: 12,
                                    color: "#fff",
                                    fontSize: 14,
                                    fontWeight: 700,
                                    cursor: predicting ? "not-allowed" : "pointer",
                                    letterSpacing: "0.04em",
                                    boxShadow: predicting ? "none" : "0 4px 24px rgba(0,200,150,0.25)",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    gap: 10,
                                }}
                            >
                                {predicting ? (
                                    <>
                                        <div
                                            style={{
                                                width: 16,
                                                height: 16,
                                                border: "2px solid rgba(255,255,255,0.3)",
                                                borderTopColor: "#fff",
                                                borderRadius: "50%",
                                                animation: "spin 0.8s linear infinite",
                                            }}
                                        />
                                        RUNNING MODEL
                                    </>
                                ) : (
                                    "PREDICT CUSTOMER SEGMENT"
                                )}
                            </button>
                        </div>

                        <div>
                            <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 12, letterSpacing: "0.08em" }}>
                                PREDICTION RESULT
                            </div>

                            {!predicted && !predicting && (
                                <div
                                    style={{
                                        background: "rgba(255,255,255,0.02)",
                                        border: "1px dashed rgba(255,255,255,0.08)",
                                        borderRadius: 14,
                                        height: 420,
                                        display: "flex",
                                        flexDirection: "column",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        gap: 12,
                                        color: "#444",
                                    }}
                                >
                                    <div style={{ fontSize: 40 }}>Target</div>
                                    <div style={{ fontSize: 13, color: "#6b7280" }}>Set RFM values and click Predict</div>
                                    <div style={{ fontSize: 11, color: "#4d5360", fontFamily: "DM Mono, monospace" }}>
                                        Random Forest model loaded from Python
                                    </div>
                                </div>
                            )}

                            {predicting && (
                                <div
                                    style={{
                                        background: "rgba(255,255,255,0.02)",
                                        border: "1px solid rgba(0,200,150,0.15)",
                                        borderRadius: 14,
                                        height: 420,
                                        display: "flex",
                                        flexDirection: "column",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        gap: 16,
                                    }}
                                >
                                    <div style={{ fontSize: 36, animation: "pulse 1s ease-in-out infinite" }}>Model</div>
                                    <div style={{ fontSize: 13, color: "#6b7280" }}>Running the backend Random Forest model</div>
                                </div>
                            )}

                            {predicted && result && seg && (
                                <div style={{ animation: "fadeUp 0.4s ease-out" }}>
                                    <div
                                        style={{
                                            background: seg.bg,
                                            border: `1px solid ${seg.border}`,
                                            borderRadius: 14,
                                            padding: 20,
                                            marginBottom: 12,
                                        }}
                                    >
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14, gap: 12 }}>
                                            <div>
                                                <div
                                                    style={{
                                                        display: "inline-block",
                                                        background: `${seg.color}20`,
                                                        border: `1px solid ${seg.color}40`,
                                                        color: seg.color,
                                                        fontSize: 9,
                                                        fontFamily: "DM Mono, monospace",
                                                        fontWeight: 700,
                                                        letterSpacing: "0.12em",
                                                        padding: "3px 8px",
                                                        borderRadius: 4,
                                                        marginBottom: 8,
                                                    }}
                                                >
                                                    {seg.badge}
                                                </div>
                                                <div style={{ fontSize: 22, fontWeight: 800, color: "#fff", letterSpacing: "-0.02em" }}>
                                                    {result.predicted}
                                                </div>
                                            </div>
                                            <div
                                                style={{
                                                    background: `${seg.color}15`,
                                                    border: `1px solid ${seg.color}30`,
                                                    borderRadius: 8,
                                                    padding: "8px 12px",
                                                    textAlign: "center",
                                                }}
                                            >
                                                <div style={{ fontSize: 24, fontWeight: 800, color: seg.color, fontFamily: "DM Mono, monospace" }}>
                                                    <AnimatedNumber value={Math.round(result.confidence * 100)} suffix="%" />
                                                </div>
                                                <div style={{ fontSize: 9, color: "#6b7280", fontFamily: "DM Mono, monospace" }}>CONFIDENCE</div>
                                            </div>
                                        </div>

                                        <p style={{ fontSize: 12, color: "#a0a6b2", lineHeight: 1.6, marginBottom: 14 }}>{seg.description}</p>

                                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                            <div style={{ width: 8, height: 8, borderRadius: "50%", background: seg.color }} />
                                            <span style={{ fontSize: 11, color: seg.color, fontFamily: "DM Mono, monospace", fontWeight: 600 }}>
                                                {seg.risk}
                                            </span>
                                        </div>
                                    </div>

                                    <div
                                        style={{
                                            background: "rgba(255,255,255,0.02)",
                                            border: "1px solid rgba(255,255,255,0.06)",
                                            borderRadius: 14,
                                            padding: "16px 18px",
                                            marginBottom: 12,
                                        }}
                                    >
                                        <div style={{ fontSize: 10, color: "#4d5360", fontFamily: "DM Mono, monospace", marginBottom: 12, letterSpacing: "0.08em" }}>
                                            PROBABILITY DISTRIBUTION
                                        </div>
                                        {Object.entries(result.probabilities)
                                            .sort((a, b) => b[1] - a[1])
                                            .map(([label, prob]) => (
                                                <ConfidenceBar
                                                    key={label}
                                                    label={label}
                                                    value={prob}
                                                    color={SEGMENTS[label].color}
                                                    isTop={label === result.predicted}
                                                />
                                            ))}
                                    </div>

                                    <div
                                        style={{
                                            background: "rgba(255,255,255,0.02)",
                                            border: "1px solid rgba(255,255,255,0.06)",
                                            borderRadius: 14,
                                            padding: "16px 18px",
                                        }}
                                    >
                                        <div style={{ fontSize: 10, color: "#4d5360", fontFamily: "DM Mono, monospace", marginBottom: 12, letterSpacing: "0.08em" }}>
                                            RECOMMENDED ACTIONS
                                        </div>
                                        {seg.actions.map((action, index) => (
                                            <div key={action} style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "flex-start" }}>
                                                <div
                                                    style={{
                                                        minWidth: 20,
                                                        height: 20,
                                                        background: `${seg.color}20`,
                                                        border: `1px solid ${seg.color}30`,
                                                        borderRadius: 6,
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent: "center",
                                                        fontSize: 10,
                                                        color: seg.color,
                                                        fontFamily: "DM Mono, monospace",
                                                        fontWeight: 700,
                                                    }}
                                                >
                                                    {index + 1}
                                                </div>
                                                <span style={{ fontSize: 12, color: "#c5cad3", lineHeight: 1.5 }}>{action}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {activeTab === "stats" && (
                <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 24px" }}>
                    <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 20, letterSpacing: "0.08em" }}>
                        TRAINING DATASET AND MODEL SNAPSHOT
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
                        {[
                            { label: "Total Customers", value: stats.total_customers, color: "#3B9EFF" },
                            { label: "Customer Segments", value: stats.classes.length, color: "#00C896" },
                            { label: "Min Recency", value: stats.recency_min, suffix: " days", color: "#F59E0B" },
                            { label: "Max Monetary", value: Math.round(stats.monetary_max), prefix: "£", color: "#FF5C5C" },
                        ].map((item) => (
                            <div
                                key={item.label}
                                style={{
                                    background: "rgba(255,255,255,0.03)",
                                    border: "1px solid rgba(255,255,255,0.07)",
                                    borderRadius: 12,
                                    padding: 16,
                                }}
                            >
                                <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 6 }}>{item.label}</div>
                                <div style={{ fontSize: 22, fontWeight: 800, color: item.color, fontFamily: "DM Mono, monospace" }}>
                                    <AnimatedNumber value={item.value} prefix={item.prefix || ""} suffix={item.suffix || ""} />
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 20 }}>
                            <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 16, letterSpacing: "0.08em" }}>
                                CUSTOMER SEGMENT BREAKDOWN
                            </div>
                            {Object.entries(stats.segment_counts).map(([segment, count]) => {
                                const pct = (count / stats.total_customers) * 100;
                                const color = SEGMENTS[segment]?.color || "#3B9EFF";
                                return (
                                    <div key={segment} style={{ marginBottom: 16 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                            <span style={{ fontSize: 12, color: "#d7dce4" }}>{segment}</span>
                                            <span style={{ fontSize: 12, color, fontFamily: "DM Mono, monospace", fontWeight: 700 }}>
                                                {count.toLocaleString()} ({pct.toFixed(1)}%)
                                            </span>
                                        </div>
                                        <div style={{ height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
                                            <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 4 }} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 20 }}>
                            <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 16, letterSpacing: "0.08em" }}>
                                RFM FEATURE RANGES
                            </div>
                            {[
                                { label: "Recency", min: stats.recency_min, max: stats.recency_max, color: "#FF5C5C", unit: "days" },
                                { label: "Frequency", min: stats.frequency_min, max: stats.frequency_max, color: "#3B9EFF", unit: "orders" },
                                { label: "Monetary", min: stats.monetary_min, max: stats.monetary_max, color: "#F59E0B", unit: "GBP" },
                            ].map((item) => (
                                <div
                                    key={item.label}
                                    style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center",
                                        padding: "12px 0",
                                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                                    }}
                                >
                                    <span style={{ fontSize: 12, color: item.color, fontFamily: "DM Mono, monospace", fontWeight: 700, width: 90 }}>
                                        {item.label}
                                    </span>
                                    <div style={{ textAlign: "right" }}>
                                        <div style={{ fontSize: 12, color: "#d7dce4" }}>Min: {item.unit === "GBP" ? `£${item.min.toFixed(2)}` : item.min}</div>
                                        <div style={{ fontSize: 11, color: "#6b7280" }}>Max: {item.unit === "GBP" ? `£${Math.round(item.max).toLocaleString()}` : item.max}</div>
                                    </div>
                                </div>
                            ))}

                            <div style={{ marginTop: 16, padding: "10px 12px", background: "rgba(0,200,150,0.06)", border: "1px solid rgba(0,200,150,0.15)", borderRadius: 8 }}>
                                <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 4 }}>SERVICE LAYER</div>
                                <div style={{ fontSize: 11, color: "#c5cad3" }}>FastAPI serves the saved Random Forest model and label encoder to the React UI.</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === "about" && (
                <div style={{ maxWidth: 760, margin: "0 auto", padding: "28px 24px" }}>
                    <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "DM Mono, monospace", marginBottom: 20, letterSpacing: "0.08em" }}>
                        PROJECT NOTES
                    </div>

                    {[
                        {
                            title: "Dataset",
                            body:
                                "Online Retail II customer transactions processed into RFM features and labeled customer segments for model training.",
                        },
                        {
                            title: "Model",
                            body:
                                "The backend loads the trained Random Forest and returns the real predicted segment plus class probabilities for the selected RFM values.",
                        },
                        {
                            title: "Frontend Architecture",
                            body:
                                "This UI is a Vite React app. The browser calls the Python API rather than trying to read the pickle file directly.",
                        },
                        {
                            title: "Run Order",
                            body:
                                "Start the API first, then run the Vite dev server. The frontend proxies /api requests to the Python service during development.",
                        },
                    ].map((item) => (
                        <div
                            key={item.title}
                            style={{
                                background: "rgba(255,255,255,0.02)",
                                border: "1px solid rgba(255,255,255,0.06)",
                                borderRadius: 12,
                                padding: "16px 18px",
                                marginBottom: 10,
                            }}
                        >
                            <div style={{ fontSize: 13, fontWeight: 700, color: "#dde2ea", marginBottom: 8 }}>{item.title}</div>
                            <p style={{ fontSize: 12, color: "#97a0ae", lineHeight: 1.7 }}>{item.body}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
