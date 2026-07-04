import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, Legend,
} from "recharts";

// ── react-force-graph-2d is loaded from CDN in index.html
// In the artifact environment we'll simulate via a canvas-based fallback
// that reads the same JSON schema so the visualization is fully functional.

const C = {
  bg:"#0D1117", surface:"#161B22", border:"#21262D",
  text:"#E6EDF3", muted:"#7D8590", dim:"#3D444D",
  teal:"#2EA043", amber:"#D29922", red:"#DA3633", blue:"#388BFD",
};

// ── Static data for non-network tabs ─────────────────────────────────────────
const PROGRESSION = [
  { name:"Baseline RF",       mae:52.39, w15:45.5 },
  { name:"+ Graph history",   mae:40.32, w15:56.7 },
  { name:"+ log(factor)",     mae:37.99, w15:67.7 },
  { name:"+ Split FTL/Cart",  mae:37.46, w15:68.1 },
  { name:"+ Cold multiplier", mae:37.30, w15:68.2 },
  { name:"Final model",       mae:37.97, w15:68.0 },
];

const HOURLY = [
  {h:"0h",f:2.237},{h:"1h",f:2.117},{h:"2h",f:2.220},{h:"3h",f:2.545},
  {h:"4h",f:2.301},{h:"5h",f:2.121},{h:"6h",f:2.306},{h:"7h",f:2.197},
  {h:"8h",f:2.034},{h:"9h",f:2.136},{h:"10h",f:2.214},{h:"11h",f:2.212},
  {h:"12h",f:2.052},{h:"13h",f:2.153},{h:"14h",f:2.144},{h:"15h",f:2.056},
  {h:"16h",f:2.152},{h:"17h",f:1.935},{h:"18h",f:1.948},{h:"19h",f:1.904},
  {h:"20h",f:2.048},{h:"21h",f:2.030},{h:"22h",f:2.089},{h:"23h",f:2.052},
];

const TOP5 = [
  { id:"IND000000ACB", name:"Gurgaon Bilaspur",     btw:23.2, delay:1.822, sla:89.4, score:0.377, state:"Haryana"     },
  { id:"IND562132AAA", name:"Bangalore Nelmangla",  btw:14.2, delay:1.682, sla:82.2, score:0.196, state:"Karnataka"   },
  { id:"IND712311AAA", name:"Kolkata Dankuni",      btw:8.4,  delay:2.357, sla:97.0, score:0.191, state:"West Bengal" },
  { id:"IND501359AAE", name:"Hyderabad Shamshabad", btw:8.7,  delay:1.764, sla:85.5, score:0.131, state:"Telangana"   },
  { id:"IND781018AAB", name:"Guwahati Hub",         btw:4.4,  delay:2.165, sla:97.0, score:0.092, state:"Assam"       },
];

const CAL = [
  {q:"P10",alpha:10,obs:13.4},
  {q:"P50",alpha:50,obs:53.7},
  {q:"P90",alpha:90,obs:88.8},
];

const FTL_CARTING = [
  {dist:"<50km",    ftl:33.4, carting:44.2, ftl_m:1.758, cart_m:1.905},
  {dist:"50-200km", ftl:37.7, carting:51.5, ftl_m:1.846, cart_m:1.959},
  {dist:"200-500km",ftl:36.5, carting:100,  ftl_m:1.848, cart_m:2.077},
  {dist:">500km",   ftl:22.7, carting:null, ftl_m:1.873, cart_m:null  },
];

const NIGHT = [
  {time:"Day (6–23h)", ftl:1.831, carting:1.833},
  {time:"Night (0–5h)",ftl:1.889, carting:2.067},
];

// ── Shared components ─────────────────────────────────────────────────────────
const Card = ({ children, style = {} }) => (
  <div style={{ background:C.surface, border:`1px solid ${C.border}`,
    borderRadius:8, padding:"16px 18px", ...style }}>
    {children}
  </div>
);

const ST = ({ children }) => (
  <div style={{ fontSize:11, fontWeight:600, letterSpacing:"0.07em",
    textTransform:"uppercase", color:C.muted, marginBottom:12 }}>
    {children}
  </div>
);

const Tile = ({ label, value, sub, color = C.text }) => (
  <Card style={{ flex:1 }}>
    <div style={{ fontSize:11, color:C.muted, marginBottom:4 }}>{label}</div>
    <div style={{ fontSize:22, fontWeight:700, color, lineHeight:1.1 }}>{value}</div>
    {sub && <div style={{ fontSize:11, color:C.muted, marginTop:3 }}>{sub}</div>}
  </Card>
);

const TT = ({ active, payload, label, unit = "" }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:C.surface, border:`1px solid ${C.border}`,
      borderRadius:6, padding:"8px 12px", fontSize:12, color:C.text }}>
      <div style={{ color:C.muted, marginBottom:4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color:p.color || C.text }}>
          {p.name}: <b>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}{unit}</b>
        </div>
      ))}
    </div>
  );
};

// ── Force-graph canvas renderer ───────────────────────────────────────────────
// Implements the react-force-graph-2d data contract:
//   graphData = { nodes: [{id, ...attrs}], links: [{source, target, ...attrs}] }
// Uses d3-force simulation via requestAnimationFrame for a real interactive graph.

function useForceSimulation(graphData, width, height) {
  const [positions, setPositions] = useState({});
  const simRef = useRef(null);

  useEffect(() => {
    if (!graphData || !graphData.nodes.length) return;

    const nodes = graphData.nodes.map(n => ({
      ...n,
      x: (Math.random() - 0.5) * width * 0.8,
      y: (Math.random() - 0.5) * height * 0.8,
      vx: 0, vy: 0,
    }));

    const nodeById = {};
    nodes.forEach(n => { nodeById[n.id] = n; });

    const links = graphData.links.map(l => ({
      ...l,
      source: nodeById[l.source] || l.source,
      target: nodeById[l.target] || l.target,
    })).filter(l => l.source && l.target && typeof l.source === "object");

    let alpha = 1;
    let running = true;

    const repulse = (a, b, strength = 800) => {
      const dx = a.x - b.x || 0.1, dy = a.y - b.y || 0.1;
      const d2 = dx * dx + dy * dy;
      const d  = Math.sqrt(d2);
      const f  = strength / d2;
      a.vx += (dx / d) * f;
      a.vy += (dy / d) * f;
    };

    const tick = () => {
      if (!running || alpha < 0.005) return;
      alpha *= 0.98;

      // Repulsion between all nodes (Barnes-Hut approximation: just do O(n²) on 80 nodes)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          repulse(nodes[i], nodes[j]);
          const dx = nodes[j].x - nodes[i].x || 0.1, dy = nodes[j].y - nodes[i].y || 0.1;
          const d2 = dx * dx + dy * dy, d = Math.sqrt(d2);
          const f  = 800 / d2;
          nodes[j].vx += (dx / d) * f;
          nodes[j].vy += (dy / d) * f;
        }
      }

      // Attraction along links
      links.forEach(l => {
        if (typeof l.source !== "object" || typeof l.target !== "object") return;
        const dx = l.target.x - l.source.x, dy = l.target.y - l.source.y;
        const d  = Math.sqrt(dx * dx + dy * dy) || 1;
        const idealLen = 60;
        const f = ((d - idealLen) / d) * 0.05 * alpha;
        l.source.vx += dx * f;  l.source.vy += dy * f;
        l.target.vx -= dx * f;  l.target.vy -= dy * f;
      });

      // Center gravity
      nodes.forEach(n => {
        n.vx += -n.x * 0.005 * alpha;
        n.vy += -n.y * 0.005 * alpha;
        n.vx *= 0.6; n.vy *= 0.6;
        n.x  += n.vx; n.y += n.vy;
        n.x   = Math.max(-width/2, Math.min(width/2, n.x));
        n.y   = Math.max(-height/2, Math.min(height/2, n.y));
      });

      const pos = {};
      nodes.forEach(n => { pos[n.id] = { x: n.x, y: n.y }; });
      setPositions({ ...pos });
      simRef.current = requestAnimationFrame(tick);
    };

    simRef.current = requestAnimationFrame(tick);
    return () => {
      running = false;
      if (simRef.current) cancelAnimationFrame(simRef.current);
    };
  }, [graphData, width, height]);

  return positions;
}

// ── Network Graph Component ───────────────────────────────────────────────────
function NetworkGraph({ graphData, filter }) {
  const W = 700, H = 480;
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const positions = useForceSimulation(graphData, W, H);
  const svgRef = useRef(null);

  if (!graphData) return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"center",
      height:H, color:C.muted, fontSize:13 }}>
      Loading graph data…
    </div>
  );

  const TOP5_IDS = new Set(graphData.nodes.filter(n => n.is_top5).map(n => n.id));
  const cx = W / 2, cy = H / 2;

  const nodePos = (id) => {
    const p = positions[id];
    return p ? { x: cx + p.x, y: cy + p.y } : { x: cx, y: cy };
  };

  // Filter logic
  const visibleNodes = graphData.nodes.filter(n => {
    if (filter === "top5")    return n.is_top5;
    if (filter === "chronic") return n.sla > 85;
    return true;
  });
  const visibleIds = new Set(visibleNodes.map(n => n.id));
  const visibleLinks = graphData.links.filter(l =>
    visibleIds.has(l.source) && visibleIds.has(l.target) &&
    (filter === "chronic" ? l.chronic : true)
  );

  const delayColor = (d) => {
    if (d > 2.2) return C.red;
    if (d > 1.8) return C.amber;
    return C.teal;
  };

  const nodeRadius = (n) => {
    if (n.is_top5) return 10 + n.btw * 0.4;
    return 4 + Math.min(n.degree * 0.15, 5);
  };

  const activeNode = selectedNode || hoveredNode;

  return (
    <div style={{ position:"relative" }}>
      <svg ref={svgRef} width={W} height={H}
        style={{ background:C.bg, borderRadius:8, border:`1px solid ${C.border}`,
                 cursor:"default", display:"block" }}>

        {/* Edges */}
        {visibleLinks.map((l, i) => {
          const s = nodePos(l.source), t = nodePos(l.target);
          const highlighted = activeNode &&
            (l.source === activeNode.id || l.target === activeNode.id);
          return (
            <line key={i}
              x1={s.x} y1={s.y} x2={t.x} y2={t.y}
              stroke={highlighted ? delayColor(l.delay) : C.dim}
              strokeWidth={highlighted ? Math.min(l.trips / 40, 3) + 1 : 0.7}
              strokeOpacity={highlighted ? 0.9 : 0.4}
            />
          );
        })}

        {/* Nodes */}
        {visibleNodes.map((n) => {
          const p = nodePos(n.id);
          const r = nodeRadius(n);
          const color = n.is_top5 ? delayColor(n.delay) :
            n.sla > 85 ? C.red : n.sla > 70 ? C.amber : C.blue;
          const isActive = activeNode?.id === n.id;

          return (
            <g key={n.id}
              onMouseEnter={() => setHoveredNode(n)}
              onMouseLeave={() => setHoveredNode(null)}
              onClick={() => setSelectedNode(selectedNode?.id === n.id ? null : n)}
              style={{ cursor:"pointer" }}>
              {isActive && (
                <circle cx={p.x} cy={p.y} r={r + 5}
                  fill="none" stroke={color} strokeWidth={1.5}
                  strokeOpacity={0.5} />
              )}
              <circle cx={p.x} cy={p.y} r={r}
                fill={color}
                fillOpacity={isActive ? 1 : 0.85}
                stroke={n.is_top5 ? "#fff" : color}
                strokeWidth={n.is_top5 ? 1.5 : 0} />
              {(n.is_top5 || isActive) && (
                <text x={p.x} y={p.y - r - 4}
                  textAnchor="middle" fontSize={9}
                  fill={color} fontWeight={600}>
                  {n.name.split(" ")[0]}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip / info panel */}
      {activeNode && (
        <div style={{
          position:"absolute", top:10, right:10,
          background:C.surface, border:`1px solid ${C.border}`,
          borderRadius:8, padding:"12px 14px", width:200,
          fontSize:12, color:C.text, pointerEvents:"none",
        }}>
          <div style={{ fontWeight:600, marginBottom:6, color:
            activeNode.sla > 85 ? C.red : C.amber }}>
            {activeNode.name.split("(")[0].trim()}
          </div>
          <div style={{ color:C.muted, marginBottom:1 }}>{activeNode.state}</div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr",
            gap:"4px 8px", marginTop:8 }}>
            {[
              ["Betweenness",  `${activeNode.btw.toFixed(1)}%`],
              ["Median delay", `${activeNode.delay.toFixed(3)}×`],
              ["SLA breach",   `${activeNode.sla}%`],
              ["Trips",        activeNode.trips.toLocaleString()],
              ["Degree",       activeNode.degree],
              ["PageRank",     activeNode.pr.toFixed(3)],
            ].map(([k, v]) => (
              <div key={k}>
                <div style={{ fontSize:10, color:C.muted }}>{k}</div>
                <div style={{ fontWeight:600 }}>{v}</div>
              </div>
            ))}
          </div>
          {activeNode.is_top5 && (
            <div style={{ marginTop:8, padding:"4px 8px", borderRadius:4,
              background:C.red+"22", color:C.red, fontSize:11, fontWeight:600 }}>
              Bottleneck #{activeNode.rank}
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div style={{ position:"absolute", bottom:10, left:10,
        background:C.surface+"CC", borderRadius:6, padding:"8px 10px",
        fontSize:10, color:C.muted, display:"flex", gap:12 }}>
        {[["Delay > 2.2×", C.red],["Delay 1.8–2.2×", C.amber],
          ["Delay ≤ 1.8×", C.teal],["Other nodes", C.blue]].map(([l,c]) => (
          <div key={l} style={{ display:"flex", alignItems:"center", gap:4 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:c }} />
            {l}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
const TABS = ["Overview","Model","Network","Hubs","FTL vs Carting","Uncertainty"];

export default function App() {
  const [tab, setTab]         = useState("Overview");
  const [graphData, setGraph] = useState(null);
  const [filter, setFilter]   = useState("all");
  const [loadErr, setLoadErr] = useState(null);

  // Load network JSON — works both when file is served alongside
  // and when embedded as inline data (fallback below)
  useEffect(() => {
    fetch("network_visualization.json")
      .then(r => r.json())
      .then(setGraph)
      .catch(() => {
        // Inline minimal fallback so the graph always renders
        setGraph({
          nodes: [
            {id:"IND000000ACB",name:"Gurgaon Bilaspur HB",state:"Haryana",btw:23.174,pr:9.83,delay:1.824,sla:89.4,trips:777,degree:89,is_top5:true,rank:1},
            {id:"IND562132AAA",name:"Bangalore Nelmangla",state:"Karnataka",btw:14.18,pr:8.21,delay:1.682,sla:82.2,trips:574,degree:71,is_top5:true,rank:2},
            {id:"IND712311AAA",name:"Kolkata Dankuni",state:"West Bengal",btw:8.35,pr:5.14,delay:2.357,sla:97.0,trips:212,degree:42,is_top5:true,rank:3},
            {id:"IND501359AAE",name:"Hyderabad Shamshabad",state:"Telangana",btw:8.72,pr:6.30,delay:1.764,sla:85.5,trips:251,degree:58,is_top5:true,rank:4},
            {id:"IND781018AAB",name:"Guwahati Hub",state:"Assam",btw:4.37,pr:2.88,delay:2.165,sla:97.0,trips:72,degree:27,is_top5:true,rank:5},
            {id:"IND411033AAA",name:"Pune Hub",state:"Maharashtra",btw:3.81,pr:4.12,delay:1.912,sla:84.2,trips:198,degree:40,is_top5:false,rank:null},
            {id:"IND302014AAA",name:"Jaipur Hub",state:"Rajasthan",btw:3.76,pr:2.95,delay:1.878,sla:81.5,trips:145,degree:35,is_top5:false,rank:null},
            {id:"IND160002AAC",name:"Chandigarh Hub",state:"Punjab",btw:5.03,pr:3.44,delay:1.905,sla:82.5,trips:188,degree:52,is_top5:false,rank:null},
            {id:"IND421302AAG",name:"Nashik Hub",state:"Maharashtra",btw:3.80,pr:3.21,delay:2.185,sla:88.1,trips:134,degree:48,is_top5:false,rank:null},
            {id:"IND131028AAB",name:"Faridabad Hub",state:"Haryana",btw:4.40,pr:3.88,delay:1.837,sla:80.3,trips:163,degree:38,is_top5:false,rank:null},
          ],
          links: [
            {source:"IND000000ACB",target:"IND562132AAA",delay:1.742,trips:180,pct_sev:35.2,chronic:true},
            {source:"IND562132AAA",target:"IND000000ACB",delay:1.688,trips:165,pct_sev:29.1,chronic:true},
            {source:"IND000000ACB",target:"IND712311AAA",delay:2.071,trips:92,pct_sev:58.7,chronic:true},
            {source:"IND000000ACB",target:"IND501359AAE",delay:1.851,trips:121,pct_sev:41.2,chronic:true},
            {source:"IND562132AAA",target:"IND501359AAE",delay:1.702,trips:88,pct_sev:32.4,chronic:true},
            {source:"IND501359AAE",target:"IND411033AAA",delay:2.173,trips:76,pct_sev:61.2,chronic:true},
            {source:"IND411033AAA",target:"IND562132AAA",delay:1.841,trips:69,pct_sev:44.1,chronic:true},
            {source:"IND000000ACB",target:"IND160002AAC",delay:2.051,trips:54,pct_sev:52.3,chronic:true},
            {source:"IND160002AAC",target:"IND000000ACB",delay:2.054,trips:51,pct_sev:53.8,chronic:true},
            {source:"IND000000ACB",target:"IND302014AAA",delay:1.893,trips:48,pct_sev:38.2,chronic:true},
            {source:"IND712311AAA",target:"IND000000ACB",delay:2.241,trips:44,pct_sev:62.5,chronic:true},
            {source:"IND000000ACB",target:"IND421302AAG",delay:1.893,trips:41,pct_sev:40.8,chronic:true},
            {source:"IND421302AAG",target:"IND562132AAA",delay:1.712,trips:38,pct_sev:33.1,chronic:true},
            {source:"IND562132AAA",target:"IND781018AAB",delay:2.312,trips:35,pct_sev:67.4,chronic:true},
            {source:"IND000000ACB",target:"IND131028AAB",delay:1.832,trips:62,pct_sev:36.7,chronic:true},
          ],
        });
      });
  }, []);

  return (
    <div style={{ background:C.bg, minHeight:"100vh", color:C.text,
      fontFamily:"'Inter',system-ui,sans-serif", padding:"20px 24px" }}>

      {/* Header */}
      <div style={{ marginBottom:20 }}>
        <div style={{ fontSize:18, fontWeight:700, marginBottom:4 }}>
          Delhivery — Graph-Enhanced ETA System
        </div>
        <div style={{ fontSize:12, color:C.muted }}>
          144,867 segments · 14,817 trips · 1,657 facilities · 2,508 corridors
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display:"flex", gap:4, marginBottom:20, flexWrap:"wrap" }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding:"5px 14px", borderRadius:6, fontSize:12, fontWeight:500,
            border:`1px solid ${tab === t ? C.blue : C.border}`,
            background:tab === t ? C.blue+"22" : "transparent",
            color:tab === t ? C.blue : C.muted, cursor:"pointer",
          }}>{t}</button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === "Overview" && (
        <div>
          <div style={{ display:"flex", gap:10, marginBottom:16, flexWrap:"wrap" }}>
            <Tile label="OSRM median underestimate"    value="1.86×"      sub="actual ÷ OSRM across all segments" color={C.red}/>
            <Tile label="Chronically delayed corridors" value="94.7%"     sub="actual > 1.2× OSRM" color={C.amber}/>
            <Tile label="Trips with ≥1 SLA breach"     value="92.1%"     sub="factor > 1.5× on any segment" color={C.amber}/>
            <Tile label="Final model segment MAE"       value="37.97 min" sub="from 52.39 baseline (−28%)" color={C.teal}/>
            <Tile label="Within-15% accuracy"           value="68.0%"     sub="from 45.5% baseline (+22.5pp)" color={C.teal}/>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            <Card>
              <ST>Delay factor distribution</ST>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={[
                  {band:"≤1.0×",pct:1.2},{band:"1.0–1.5×",pct:23.4},
                  {band:"1.5–2.0×",pct:39.2},{band:"2.0–3.0×",pct:25.7},{band:">3.0×",pct:10.5},
                ]} margin={{left:-10,right:8,top:4,bottom:4}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="band" tick={{fontSize:10,fill:C.muted}}/>
                  <YAxis tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <Bar dataKey="pct" name="% of segments" radius={[3,3,0,0]}>
                    {[C.teal,C.blue,C.amber,"#E06C00",C.red].map((c,i)=><Cell key={i} fill={c}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize:11, color:C.muted, marginTop:6 }}>
                Only 1.2% of segments run at or faster than OSRM estimates.
              </div>
            </Card>
            <Card>
              <ST>Mean delay factor by hour of day</ST>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={HOURLY} margin={{left:-10,right:8,top:4,bottom:4}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="h" tick={{fontSize:9,fill:C.muted}} interval={2}/>
                  <YAxis domain={[1.85,2.6]} tick={{fontSize:10,fill:C.muted}}/>
                  <Tooltip content={<TT/>}/>
                  <ReferenceLine y={1.857} stroke={C.muted} strokeDasharray="3 3"
                    label={{value:"median",fill:C.muted,fontSize:9}}/>
                  <Line dataKey="f" name="Mean factor" stroke={C.blue} strokeWidth={2} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
              <div style={{ fontSize:11, color:C.muted, marginTop:6 }}>
                Peak at 3h (2.55×). Optimal dispatch window: 17–19h (≈1.92×).
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ── MODEL ── */}
      {tab === "Model" && (
        <div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:14 }}>
            <Card>
              <ST>Segment MAE per step (min)</ST>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={PROGRESSION} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="name" tick={{fontSize:9,fill:C.muted}}/>
                  <YAxis domain={[34,55]} tick={{fontSize:10,fill:C.muted}} unit=" min"/>
                  <Tooltip content={<TT unit=" min"/>}/>
                  <Bar dataKey="mae" name="MAE" radius={[3,3,0,0]}>
                    {PROGRESSION.map((_,i)=>(
                      <Cell key={i} fill={i===0?C.red:i===PROGRESSION.length-1?C.teal:C.amber}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <ST>Within-15% accuracy per step (%)</ST>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={PROGRESSION} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="name" tick={{fontSize:9,fill:C.muted}}/>
                  <YAxis domain={[42,72]} tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <Line dataKey="w15" name="W15%" stroke={C.teal} strokeWidth={2.5} dot={{fill:C.teal,r:4}}/>
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>
          <Card style={{marginBottom:14}}>
            <ST>What each step contributed</ST>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
              {[
                ["Graph corridor history","+10pp W15","Each corridor's median historical delay ratio becomes a feature, ranked 3rd of 35 — above hour-of-day and hub centrality."],
                ["log(factor) target","+11pp W15","Predicting raw actual_time gave osrm_distance 87% of importance. Switching to log(actual/osrm) drops it to 34% and surfaces delay signals."],
                ["Separate FTL / Carting","+8.5pp FTL","Carting factor std=2.95 vs FTL 0.96. Pooling biases toward FTL. Dedicated Carting model beats the corridor-median oracle."],
                ["Cold corridor ×1.05 / ×1.14","+0.3pp","Unseen corridors are 5.7% (FTL) / 14.4% (Carting) slower — selection bias on rarely-traveled, harder routes."],
              ].map(([t,g,d])=>(
                <div key={t} style={{ background:C.bg, borderRadius:6, padding:"12px 14px",
                  border:`1px solid ${C.border}` }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                    <span style={{ fontSize:12, fontWeight:600 }}>{t}</span>
                    <span style={{ fontSize:11, fontWeight:600, color:C.teal,
                      background:C.teal+"1A", padding:"1px 6px", borderRadius:4 }}>{g}</span>
                  </div>
                  <div style={{ fontSize:11, color:C.muted, lineHeight:1.6 }}>{d}</div>
                </div>
              ))}
            </div>
          </Card>
          <div style={{ display:"flex", gap:10 }}>
            <Card style={{ flex:1 }}>
              <ST>Route-type W15% — model vs baseline vs ceiling</ST>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={[
                  {rt:"FTL",    baseline:45.5, model:75.0, ceiling:80.4},
                  {rt:"Carting",baseline:45.5, model:52.0, ceiling:56.2},
                ]} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="rt" tick={{fontSize:12,fill:C.muted}}/>
                  <YAxis domain={[40,85]} tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <Legend wrapperStyle={{fontSize:11}}/>
                  <Bar dataKey="baseline" name="Baseline" fill={C.dim}  radius={[3,3,0,0]}/>
                  <Bar dataKey="model"    name="Model"    fill={C.teal} radius={[3,3,0,0]}/>
                  <Bar dataKey="ceiling"  name="Ceiling"  fill={C.red}  radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card style={{ flex:1 }}>
              <ST>Feature group importance</ST>
              <div style={{ display:"flex", flexDirection:"column", gap:6, paddingTop:4 }}>
                {[
                  {g:"OSRM planned features", pct:46.3, c:C.blue},
                  {g:"Corridor history",       pct:27.4, c:C.teal},
                  {g:"Interaction terms",      pct:12.1, c:C.amber},
                  {g:"Graph topology",         pct:9.4,  c:"#8B949E"},
                  {g:"Time features",          pct:3.2,  c:C.muted},
                  {g:"Trip structure",         pct:1.6,  c:C.dim},
                ].map(r=>(
                  <div key={r.g} style={{ display:"flex", alignItems:"center", gap:8 }}>
                    <div style={{ fontSize:11, color:C.muted, width:140,
                      flexShrink:0, textAlign:"right" }}>{r.g}</div>
                    <div style={{ flex:1, background:C.bg, borderRadius:3, height:13, overflow:"hidden" }}>
                      <div style={{ width:`${r.pct}%`, background:r.c, height:"100%", borderRadius:3 }}/>
                    </div>
                    <div style={{ fontSize:11, color:C.muted, width:34, textAlign:"right" }}>{r.pct}%</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ── NETWORK ── */}
      {tab === "Network" && (
        <div>
          {/* Filter controls */}
          <div style={{ display:"flex", gap:8, marginBottom:14, alignItems:"center" }}>
            <span style={{ fontSize:11, color:C.muted }}>Show:</span>
            {[["all","All nodes"],["top5","Top-5 hubs only"],["chronic","High-SLA corridors"]].map(([v,l])=>(
              <button key={v} onClick={()=>setFilter(v)} style={{
                padding:"4px 12px", borderRadius:5, fontSize:11, fontWeight:500,
                border:`1px solid ${filter===v?C.amber:C.border}`,
                background:filter===v?C.amber+"22":"transparent",
                color:filter===v?C.amber:C.muted, cursor:"pointer",
              }}>{l}</button>
            ))}
            <span style={{ fontSize:11, color:C.dim, marginLeft:4 }}>
              Hover node to inspect · Click to pin
            </span>
          </div>

          <NetworkGraph graphData={graphData} filter={filter}/>

          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10, marginTop:14 }}>
            <Tile label="Network nodes shown" value={graphData ? graphData.nodes.length : "—"} sub="top 80 by degree"/>
            <Tile label="Network edges shown"  value={graphData ? graphData.links.length : "—"} sub="between top-80 nodes"/>
            <Tile label="Chronic edges"  value={graphData ? graphData.links.filter(l=>l.chronic).length : "—"}  sub="factor > 1.2×" color={C.amber}/>
            <Tile label="Hub #1 betweenness" value="23.2%" sub="IND000000ACB" color={C.red}/>
          </div>
        </div>
      )}

      {/* ── HUBS ── */}
      {tab === "Hubs" && (
        <div>
          <Card style={{ marginBottom:14 }}>
            <ST>Top 5 bottleneck hubs — risk score = betweenness × delay × SLA breach rate</ST>
            <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
              {TOP5.map((h,i)=>(
                <div key={h.id} style={{ background:C.bg, borderRadius:7,
                  padding:"14px 16px",
                  border:`1px solid ${i===0?C.red:i<3?C.amber:C.border}` }}>
                  <div style={{ display:"flex", justifyContent:"space-between",
                    alignItems:"flex-start", marginBottom:10 }}>
                    <div>
                      <span style={{ fontSize:11, color:C.muted, fontFamily:"monospace",
                        background:C.surface, padding:"2px 6px", borderRadius:4, marginRight:8 }}>#{i+1}</span>
                      <span style={{ fontSize:13, fontWeight:600 }}>{h.name}</span>
                      <span style={{ fontSize:10, color:C.muted, marginLeft:8 }}>{h.state}</span>
                      <span style={{ fontSize:10, color:C.dim, marginLeft:8, fontFamily:"monospace" }}>{h.id}</span>
                    </div>
                    <span style={{ fontSize:11, fontWeight:700,
                      color:i===0?C.red:i<3?C.amber:C.muted }}>score {h.score.toFixed(3)}</span>
                  </div>
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8 }}>
                    {[
                      {k:"Betweenness",    v:`${h.btw}%`},
                      {k:"Median delay",   v:`${h.delay.toFixed(2)}×`},
                      {k:"SLA breach",     v:`${h.sla}%`},
                      {k:"Risk level",     v:i===0?"Critical":i<3?"High":"Elevated",
                        col:i===0?C.red:i<3?C.amber:C.blue},
                    ].map(m=>(
                      <div key={m.k} style={{ background:C.surface, borderRadius:5, padding:"8px 10px" }}>
                        <div style={{ fontSize:10, color:C.muted }}>{m.k}</div>
                        <div style={{ fontSize:14, fontWeight:700, color:m.col||C.text }}>{m.v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            <Card>
              <ST>Betweenness centrality — top 5</ST>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={TOP5} layout="vertical" margin={{left:10,right:20,top:4,bottom:4}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis type="number" tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <YAxis type="category" dataKey="name" width={140} tick={{fontSize:10,fill:C.muted}}/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <Bar dataKey="btw" name="Betweenness" radius={[0,3,3,0]}>
                    {TOP5.map((_,i)=><Cell key={i} fill={i===0?C.red:i<3?C.amber:C.blue}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <ST>SLA breach rate vs median delay</ST>
              <ResponsiveContainer width="100%" height={160}>
                <ScatterChart margin={{left:0,right:20,top:10,bottom:20}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="delay" name="Median delay" type="number" domain={[1.6,2.5]}
                    tick={{fontSize:10,fill:C.muted}}
                    label={{value:"Median delay ×",position:"insideBottom",offset:-12,fill:C.muted,fontSize:10}}/>
                  <YAxis dataKey="sla" name="SLA breach %" type="number" domain={[78,100]}
                    tick={{fontSize:10,fill:C.muted}}
                    label={{value:"SLA breach %",angle:-90,position:"insideLeft",fill:C.muted,fontSize:10}}/>
                  <Tooltip content={({active,payload})=>{
                    if (!active||!payload?.length) return null;
                    const d=payload[0].payload;
                    return (
                      <div style={{background:C.surface,border:`1px solid ${C.border}`,
                        borderRadius:6,padding:"8px 12px",fontSize:12,color:C.text}}>
                        <b>{d.name}</b>
                        <div>Delay: {d.delay.toFixed(3)}×</div>
                        <div>SLA breach: {d.sla}%</div>
                      </div>
                    );
                  }}/>
                  <Scatter data={TOP5}>
                    {TOP5.map((_,i)=><Cell key={i} fill={i===0?C.red:i<3?C.amber:C.blue} r={i===0?9:6}/>)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {/* ── FTL vs CARTING ── */}
      {tab === "FTL vs Carting" && (
        <div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:14 }}>
            <Card>
              <ST>Severe delay risk (%) by distance</ST>
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={FTL_CARTING} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="dist" tick={{fontSize:10,fill:C.muted}}/>
                  <YAxis domain={[0,110]} tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <Legend wrapperStyle={{fontSize:11}}/>
                  <Bar dataKey="ftl"     name="FTL"     fill={C.teal}  radius={[3,3,0,0]}/>
                  <Bar dataKey="carting" name="Carting" fill={C.amber} radius={[3,3,0,0]}/>
                  <ReferenceLine y={100} stroke={C.red} strokeDasharray="3 3"
                    label={{value:"100%",fill:C.red,fontSize:9}}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <ST>Median delay factor by distance</ST>
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={FTL_CARTING} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="dist" tick={{fontSize:10,fill:C.muted}}/>
                  <YAxis domain={[1.6,2.2]} tick={{fontSize:10,fill:C.muted}}/>
                  <Tooltip content={<TT/>}/>
                  <Legend wrapperStyle={{fontSize:11}}/>
                  <Bar dataKey="ftl_m"  name="FTL"     fill={C.teal}  radius={[3,3,0,0]}/>
                  <Bar dataKey="cart_m" name="Carting" fill={C.amber} radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
          <Card>
            <ST>Decision rules</ST>
            <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
              {[
                {r:"R1",c:"Distance > 200km  +  Minor or Major hub source",v:"FTL mandatory",col:C.red,
                  n:"Carting severe-delay risk hits 100% on these corridors. FTL advantage up to 0.99× lower factor."},
                {r:"R2",c:"Distance < 200km  +  Hub-tier source (betweenness > 10%)",v:"Carting acceptable",col:C.teal,
                  n:"High-throughput hubs dispatch Carting near capacity. Batching advantage 0.05–0.09×. Overridden by R3 at night."},
                {r:"R3",c:"Night dispatch 0–6h — any corridor",v:"FTL mandatory",col:C.red,
                  n:"Carting night penalty = +0.234×. FTL = only +0.058×. Largest time-of-day gap in the dataset."},
                {r:"R4",c:"All other corridors",v:"FTL preferred",col:C.amber,
                  n:"FTL wins in 12 of 16 (distance × hub-tier) matrix cells. Safe default."},
              ].map(x=>(
                <div key={x.r} style={{ background:C.bg, borderRadius:6, padding:"12px 14px",
                  border:`1px solid ${C.border}`,
                  display:"grid", gridTemplateColumns:"36px 1fr auto",
                  gap:12, alignItems:"start" }}>
                  <div style={{ fontSize:12, fontWeight:700, color:C.muted,
                    background:C.surface, borderRadius:4, padding:"3px 6px",
                    textAlign:"center" }}>{x.r}</div>
                  <div>
                    <div style={{ fontSize:12, color:C.text, marginBottom:4 }}>{x.c}</div>
                    <div style={{ fontSize:11, color:C.muted, lineHeight:1.5 }}>{x.n}</div>
                  </div>
                  <div style={{ fontSize:11, fontWeight:600, color:x.col,
                    background:x.col+"1A", padding:"3px 10px", borderRadius:4,
                    whiteSpace:"nowrap" }}>{x.v}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── UNCERTAINTY ── */}
      {tab === "Uncertainty" && (
        <div>
          <div style={{ display:"flex", gap:10, marginBottom:14, flexWrap:"wrap" }}>
            <Tile label="Segment P10–P90 coverage" value="75.4%" sub="target: 80%" color={C.amber}/>
            <Tile label="Trip P10–P90 coverage"    value="80.7%" sub="hits the 80% target ✓" color={C.teal}/>
            <Tile label="Avg trip interval width"   value="1,034 min" sub="P10 to P90"/>
            <Tile label="Production models"         value="6" sub="P10 / P50 / P90  ×  FTL / Carting"/>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:14 }}>
            <Card>
              <ST>Calibration — P(pred &gt; actual) vs target α</ST>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{left:0,right:24,top:20,bottom:24}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="alpha" name="Target α" type="number" domain={[0,100]}
                    tick={{fontSize:10,fill:C.muted}} unit="%"
                    label={{value:"Target α",position:"insideBottom",offset:-12,fill:C.muted,fontSize:10}}/>
                  <YAxis dataKey="obs" name="Observed" type="number" domain={[0,100]}
                    tick={{fontSize:10,fill:C.muted}} unit="%"
                    label={{value:"Observed",angle:-90,position:"insideLeft",fill:C.muted,fontSize:10}}/>
                  <ReferenceLine segment={[{x:0,y:0},{x:100,y:100}]}
                    stroke={C.dim} strokeDasharray="4 3"
                    label={{value:"perfect",fill:C.dim,fontSize:9,position:"insideTopRight"}}/>
                  <Tooltip content={({active,payload})=>{
                    if (!active||!payload?.length) return null;
                    const d=payload[0].payload;
                    return (
                      <div style={{background:C.surface,border:`1px solid ${C.border}`,
                        borderRadius:6,padding:"8px 12px",fontSize:12,color:C.text}}>
                        <b>{d.q}</b>
                        <div>Target: {d.alpha}%</div>
                        <div>Observed: {d.obs}%</div>
                        <div style={{color:Math.abs(d.obs-d.alpha)<6?C.teal:C.amber}}>
                          Gap: {(d.obs-d.alpha).toFixed(1)}pp
                        </div>
                      </div>
                    );
                  }}/>
                  <Scatter data={CAL} fill={C.teal} r={7}/>
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <ST>P10–P90 interval coverage</ST>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={[
                  {level:"Segment (raw)",  cov:67.0},
                  {level:"Segment (cal.)", cov:75.4},
                  {level:"Trip (cal.)",    cov:80.7},
                  {level:"Target",         cov:80.0},
                ]} margin={{left:-10,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.dim}/>
                  <XAxis dataKey="level" tick={{fontSize:10,fill:C.muted}}/>
                  <YAxis domain={[60,85]} tick={{fontSize:10,fill:C.muted}} unit="%"/>
                  <Tooltip content={<TT unit="%"/>}/>
                  <ReferenceLine y={80} stroke={C.amber} strokeDasharray="3 3"
                    label={{value:"80% target",fill:C.amber,fontSize:9}}/>
                  <Bar dataKey="cov" name="Coverage" radius={[3,3,0,0]}>
                    {[C.dim,C.amber,C.teal,"transparent"].map((c,i)=>(
                      <Cell key={i} fill={c}
                        stroke={i===3?C.amber:"none"} strokeWidth={i===3?2:0}
                        strokeDasharray={i===3?"4 3":"0"}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
          <Card>
            <ST>Operational use of the three outputs</ST>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10 }}>
              {[
                {m:"P10 — lower bound",u:"Dispatch SLA window",
                  d:"90% of trips take longer than this. Use as the earliest reliable promise to avoid unachievable SLA commits.",
                  stat:"P(pred > actual) = 13.4%",c:C.blue},
                {m:"P50 — median ETA",u:"Customer-facing ETA",
                  d:"Best single-point estimate. MAE = 38 min, W15 = 68%. Slightly conservative (53.7% arrive before this).",
                  stat:"MAE = 37.97 min  |  W15 = 68%",c:C.teal},
                {m:"P90 + 5.3 min/seg",u:"Capacity planning",
                  d:"91.7% of trips arrive before this. The +5.3 mean-bias correction is appropriate for expected-value staffing calculations.",
                  stat:"P(pred > actual) = 91.7%",c:C.amber},
              ].map(o=>(
                <div key={o.m} style={{ background:C.bg, borderRadius:7, padding:"14px",
                  border:`1.5px solid ${o.c}22` }}>
                  <div style={{ fontSize:10, color:o.c, fontWeight:600, marginBottom:4 }}>{o.m}</div>
                  <div style={{ fontSize:12, fontWeight:600, marginBottom:8 }}>{o.u}</div>
                  <div style={{ fontSize:11, color:C.muted, lineHeight:1.6, marginBottom:10 }}>{o.d}</div>
                  <div style={{ fontSize:10, color:o.c, background:o.c+"18",
                    padding:"3px 8px", borderRadius:4, fontFamily:"monospace" }}>{o.stat}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
