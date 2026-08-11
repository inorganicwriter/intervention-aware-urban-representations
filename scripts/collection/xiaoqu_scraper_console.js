/**
 * Lianjia Xiaoqu Scraper — Browser Console Script (single-city + auto-resume)
 *
 * Usage:
 *   1. Open https://{city}.lianjia.com/xiaoqu/ in your browser (logged in)
 *   2. F12 -> Console, paste this entire script, press Enter
 *   3. It scrapes all pages for THIS city, downloads CSV, then navigates to the
 *      NEXT city's xiaoqu page. On the new page, just paste again — it resumes
 *      from where it left off (progress stored in localStorage).
 *
 * To reset progress: localStorage.removeItem("lianjia_xiaoqu_progress")
 */

(async function scrapeXiaoqu() {
    "use strict";

    const LS_KEY = "lianjia_xiaoqu_progress";

    // All 44 cities. host=null means no lianjia subdomain.
    const CITY_LIST = [
        { key: "beijing",    host: "bj",       name: "北京" },
        { key: "shanghai",   host: "sh",       name: "上海" },
        { key: "guangzhou",  host: "gz",       name: "广州" },
        { key: "shenzhen",   host: "sz",       name: "深圳" },
        { key: "chengdu",    host: "cd",       name: "成都" },
        { key: "hangzhou",   host: "hz",       name: "杭州" },
        { key: "wuhan",      host: "wh",       name: "武汉" },
        { key: "nanjing",    host: "nj",       name: "南京" },
        { key: "tianjin",    host: "tj",       name: "天津" },
        { key: "chongqing",  host: "cq",       name: "重庆" },
        { key: "suzhou",     host: "su",       name: "苏州" },
        { key: "xian",       host: "xa",       name: "西安" },
        { key: "changsha",   host: "cs",       name: "长沙" },
        { key: "dalian",     host: "dl",       name: "大连" },
        { key: "shenyang",   host: "sy",       name: "沈阳" },
        { key: "qingdao",    host: "qd",       name: "青岛" },
        { key: "jinan",      host: "jn",       name: "济南" },
        { key: "foshan",     host: "fs",       name: "佛山" },
        { key: "dongguan",   host: "dg",       name: "东莞" },
        { key: "xiamen",     host: "xm",       name: "厦门" },
        { key: "hefei",      host: "hf",       name: "合肥" },
        { key: "zhengzhou",  host: "zz",       name: "郑州" },
        { key: "kunming",    host: "km",       name: "昆明" },
        { key: "fuzhou",     host: "fz",       name: "福州" },
        { key: "nanning",    host: "nn",       name: "南宁" },
        { key: "wuxi",       host: "wx",       name: "无锡" },
        { key: "ningbo",     host: "nb",       name: "宁波" },
        { key: "changchun",  host: "cc",       name: "长春" },
        { key: "guiyang",    host: "gy",       name: "贵阳" },
        { key: "shijiazhuang", host: "sjz",    name: "石家庄" },
        { key: "harbin",     host: "hrb",      name: "哈尔滨" },
        { key: "taiyuan",    host: "ty",       name: "太原" },
        { key: "nanchang",   host: "nc",       name: "南昌" },
        { key: "lanzhou",    host: "lz",       name: "兰州" },
        { key: "hohhot",     host: "hhht",     name: "呼和浩特" },
        { key: "urumqi",     host: null,       name: "乌鲁木齐" },
        { key: "wenzhou",    host: null,       name: "温州" },
        { key: "xuzhou",     host: null,       name: "徐州" },
        { key: "jinhua",     host: null,       name: "金华" },
        { key: "shaoxing",   host: null,       name: "绍兴" },
        { key: "taizhou",    host: null,       name: "台州" },
        { key: "luoyang",    host: null,       name: "洛阳" },
        { key: "nantong",    host: null,       name: "南通" },
        { key: "changzhou",  host: null,       name: "常州" },
    ];

    // ---- helpers ----
    const progress = loadProgress();
    const curInfo = detectCurCity();

    function loadProgress() {
        try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
    }
    function saveProgress(p) { localStorage.setItem(LS_KEY, JSON.stringify(p)); }
    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    function detectCurCity() {
        const host = window.location.hostname;
        return CITY_LIST.find(c => c.host && host.startsWith(c.host + ".")) || null;
    }

    function parsePositionInfo(posEl) {
        let buildTypes = "";
        let buildYears = "";
        if (!posEl) return { buildTypes, buildYears };
        const icons = posEl.querySelectorAll("span");
        let txt = posEl.textContent;
        icons.forEach(s => { txt = txt.replace(s.textContent, ""); });
        txt = txt.trim();
        const yrM = txt.match(/(\d{4}(?:[\u2014\-]+\d{4})?\u5e74)/);
        if (yrM) { buildYears = yrM[1]; txt = txt.replace(yrM[1], ""); }
        const KW = ["塔楼", "板楼", "塔板结合", "平房", "板塔结合"];
        const parts = txt.split("/").map(p => p.trim()).filter(p => KW.some(k => p.includes(k)));
        if (parts.length) buildTypes = parts.join("/");
        return { buildTypes, buildYears };
    }

    function parsePage(doc, ck) {
        const rows = [];
        doc.querySelectorAll("li.xiaoquListItem").forEach(li => {
            const n = (li.querySelector(".title a")?.textContent || "").trim();
            if (!n) return;
            const p = li.querySelector(".totalPrice span")?.textContent.trim() || "";
            const pm = p.match(/([\d.]+)/);
            const { buildTypes, buildYears } = parsePositionInfo(li.querySelector(".positionInfo"));
            const metros = [];
            li.querySelectorAll(".tagList span").forEach(s => {
                const t = s.textContent.trim();
                if (t.includes("地铁") || t.includes("号线")) metros.push(t);
            });
            rows.push({
                city: ck, name: n,
                unit_price: pm ? pm[1] : "",
                district: li.querySelector("a.district")?.textContent.trim() || "",
                bizcircle: li.querySelector("a.bizcircle")?.textContent.trim() || "",
                build_types: buildTypes, build_years: buildYears,
                metro: metros.join("; "),
            });
        });
        return rows;
    }

    async function fetchPage(pageNum) {
        const url = pageNum === 1
            ? "/xiaoqu/?from=rec"
            : `/xiaoqu/pg${pageNum}/`;
        const resp = await fetch(url, { credentials: "same-origin" });
        const html = await resp.text();
        return new DOMParser().parseFromString(html, "text/html");
    }

    function getPagination(doc) {
        const box = doc.querySelector(".house-lst-page-box");
        if (!box) return { total: 1, cur: 1 };
        try {
            const d = JSON.parse(box.getAttribute("page-data"));
            return { total: d.totalPage || 1, cur: d.curPage || 1 };
        } catch { return { total: 1, cur: 1 }; }
    }

    function downloadCSV(rows, cityKey) {
        if (!rows.length) return;
        const H = Object.keys(rows[0]);
        const lines = [H.join(",")];
        rows.forEach(r => lines.push(H.map(h => '"' + String(r[h] ?? "").replace(/"/g, '""') + '"').join(",")));
        const b = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(b);
        a.download = `${cityKey}_lianjia_xiaoqu.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
        console.log(`  [SAVED] ${a.download}  (${rows.length} rows)`);
    }

    // ---- find next city to scrape ----
    function findNext(idx) {
        for (let i = idx + 1; i < CITY_LIST.length; i++) {
            const c = CITY_LIST[i];
            if (!c.host) continue;
            if (progress[c.key] === "done") continue;
            return c;
        }
        return null;
    }

    function showProgress() {
        const done = Object.values(progress).filter(v => v === "done" || v === "no_lianjia").length;
        const nolj = Object.values(progress).filter(v => v === "no_lianjia").length;
        console.log(`Progress: ${done}/${CITY_LIST.length} done (${nolj} no lianjia)`);
        const pending = CITY_LIST.filter(c => c.host && progress[c.key] !== "done");
        if (pending.length) {
            console.log("Remaining:", pending.map(c => c.name).join(", "));
        } else {
            console.log("ALL DONE! Clear progress: localStorage.removeItem('" + LS_KEY + "')");
        }
    }

    // ===================== MAIN =====================
    if (!curInfo) {
        console.error("Could not detect current city. Please visit a lianjia.com/xiaoqu/ page first.");
        return;
    }

    const { key: ck, host: sub, name: cname } = curInfo;
    console.log(`\n=== ${cname} (${ck}) ===`);
    showProgress();

    if (progress[ck] === "done") {
        console.log(`  Already done. Skipping.`);
        const next = findNext(CITY_LIST.findIndex(c => c.key === ck));
        if (next) {
            console.log(`  Navigating to ${next.name}...`);
            window.location.href = `https://${next.host}.lianjia.com/xiaoqu/`;
        } else {
            console.log("  All cities complete!");
        }
        return;
    }

    // ---- scrap city ----
    let doc = document;
    const banner = doc.querySelector(".clearBtn a");
    if (banner) {
        console.log("  On recommended view, fetching full list...");
        doc = await fetchPage(1);
    }

    const { total } = getPagination(doc);
    console.log(`  ${total} pages`);

    const allRows = [];
    for (let pg = 1; pg <= total; pg++) {
        if (pg > 1) {
            await sleep(600 + Math.random() * 900);
            try { doc = await fetchPage(pg); }
            catch {
                console.warn(`  Page ${pg}: fetch failed, retrying...`);
                await sleep(2000);
                doc = await fetchPage(pg);
            }
        }
        const recs = parsePage(doc, ck);
        allRows.push(...recs);
        if (pg % 20 === 0 || pg === total) {
            console.log(`  ${pg}/${total}  |  ${allRows.length} collected`);
            saveProgress({ ...progress, [ck]: `pg${pg}` });
        }
    }

    if (allRows.length) {
        downloadCSV(allRows, ck);
    } else {
        console.warn("  No data collected.");
    }

    saveProgress({ ...progress, [ck]: "done" });
    showProgress();

    // Navigate to next city
    const idx = CITY_LIST.findIndex(c => c.key === ck);
    const next = findNext(idx);
    if (next) {
        console.log(`\n  >>> Navigating to ${next.name}...`);
        console.log("  >>> On the new page, paste this script again to continue.");
        await sleep(1500);
        window.location.href = `https://${next.host}.lianjia.com/xiaoqu/`;
    } else {
        // Mark no-lianjia cities too
        for (const c of CITY_LIST) {
            if (!c.host && !progress[c.key]) saveProgress({ ...progress, [c.key]: "no_lianjia" });
        }
        console.log("\n  ALL COMPLETE!");
    }
})();
