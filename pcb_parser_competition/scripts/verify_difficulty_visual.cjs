const { chromium } = require('C:/Users/Pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '..');
(async () => {
  const executablePath = ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(file => fs.existsSync(file));
  const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 800, height: 1500 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('file:///' + path.join(root, 'runs/s3_component_text_quick_dev/0001/difficulty-preview.html').replaceAll('\\', '/'));
    const frame = page.frames().find(item => item.parentFrame());
    await frame.locator('#pcb-difficulties-0001').waitFor();
    await frame.waitForFunction(() => Number(document.querySelector('.original-image').getAttribute('width')) > 0);
    await page.screenshot({ path: path.join(root, 'runs/s3_component_text_quick_dev/0001/difficulty-preview-check.png'), fullPage: true });
    await frame.locator('button[data-id="3a"]').click();
    const title = await frame.locator('.detail-title').innerText();
    if (!title.includes('TP1') || errors.length) throw new Error(JSON.stringify({ title, errors }));
    await page.setViewportSize({ width: 360, height: 1500 });
    const overflow = await frame.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    if (overflow) throw new Error('Narrow layout overflow');
    console.log(JSON.stringify({ status: 'passed', errors, afterClick: title, narrowOverflow: overflow }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
