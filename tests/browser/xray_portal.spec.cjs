/* Standalone browser regression. All API and image responses are synthetic. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createRequire} = require('node:module');

const repository = path.resolve(__dirname, '../..');
const portal = path.join(repository, 'apps/xray_portal');
const requireAssets = createRequire(path.join(portal, 'static/package.json'));
const {chromium} = requireAssets('playwright');

async function main() {
  const artifacts = fs.mkdtempSync(path.join(os.tmpdir(), 'xray-portal-browser-'));
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  const browser = await chromium.launch({
    ...(executablePath ? {executablePath} : {}),
    headless: true,
    args: ['--disable-gpu'],
  });
  try {
    let admin = false;
    let saved = null;
    let conditionalResponses = 0;
    const errors = [];
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    page.on('pageerror', error => errors.push(error.message));
    const images = [];
    const pairs = [];
    for (let slice = 0; slice < 4; slice++) {
      for (const role of ['GT', 'LQ']) {
        images.push({
          id: role + slice, role, workpiece: 'SYNTHETIC-001', scan_group: 'XY', axis: 'XY',
          condition: role === 'GT' ? 'Reference condition' : 'Noisy condition',
          filename: `data_${slice}.tif`, slice_id: slice, width: 1518, height: 1518,
          dtype: 'uint16', min: 100, max: 60000, mean: 20000,
        });
      }
      pairs.push({
        id: 'pair' + slice, gt_id: 'GT' + slice, lq_id: 'LQ' + slice,
        group: 'SYNTHETIC-001', slice_id: slice, status: 'candidate',
      });
    }
    const batch = {
      updated_at: 1, id: 'batch1', title: 'Synthetic condition ZIPs',
      workpiece: 'SYNTHETIC-001', scan_group: 'XY', uploader: 'Synthetic test',
      status: 'awaiting_review',
      files: [{
        id: 'file1', filename: 'synthetic_condition.zip', role: 'LQ', condition: 'Noisy condition',
        size: 5000000000, status: 'received',
      }],
      report: {images, candidate_pairs: pairs, issues: [{severity: 'warning', message: 'Confirm alignment.'}]},
      events: [],
    };
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      assert.equal(url.origin, 'http://portal.test', 'Browser assets must not contact a CDN or live service');
      const pathname = url.pathname;
      let data;
      if (pathname === '/api/session') data = {admin, batch_ids: ['batch1'], csrf_token: 'synthetic-csrf'};
      else if (pathname === '/api/batches') data = [{id: batch.id, title: batch.title, status: batch.status}];
      else if (pathname === '/api/admin/login') {admin = true; data = {ok: true};}
      else if (pathname === '/api/batches/batch1/review') {
        saved = route.request().postDataJSON(); batch.decisions = saved; batch.updated_at++;
        data = {ok: true};
      } else if (pathname === '/api/batches/batch1') {
        if (url.searchParams.get('after') === String(batch.updated_at)) {
          conditionalResponses++; data = {unchanged: true};
        } else data = batch;
      } else if (pathname.includes('/preview/')) {
        return route.fulfill({
          status: 200, contentType: 'image/svg+xml',
          body: '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
            + '<rect width="300" height="300" fill="#303941"/>'
            + '<circle cx="150" cy="150" r="90" fill="#adbac3"/>'
            + '<circle cx="150" cy="150" r="50" fill="#4b555f"/></svg>',
        });
      }
      if (data !== undefined) return route.fulfill({json: data});
      assert(pathname === '/' || pathname.startsWith('/static/'), `Unexpected request: ${pathname}`);
      return route.fulfill({path: pathname === '/' ? path.join(portal, 'templates/index.html') : portal + pathname});
    });

    await page.goto('http://portal.test/');
    await page.getByRole('button', {name: 'Synthetic condition ZIPs Awaiting Review'}).click();
    await page.locator('#batch-title').waitFor();
    assert(await page.locator('#review-section').isHidden());
    await page.locator('#admin-open').click();
    await page.locator('#admin-form input').fill('synthetic-only');
    await page.locator('#admin-form button.primary').click();
    await page.locator('#review-section').waitFor({state: 'visible'});
    assert.equal(await page.locator('.pair-card').count(), 4);
    assert.equal(await page.locator('.pair-card').first().locator('[data-field=gt_id] option').count(), 2,
      'Each GT dropdown should offer only its compatible reference plus the empty choice');
    await page.locator('#split-mode').selectOption('spatial');
    await page.locator('#range-start').fill('0');
    await page.locator('#range-end').fill('1');
    await page.locator('#range-split').selectOption('train');
    await page.locator('#apply-range').click();
    assert.equal(await page.locator('.pair-card').nth(0).locator('[data-field=split]').inputValue(), 'train');
    assert.equal(await page.locator('.pair-card').nth(2).locator('[data-field=split]').inputValue(), 'exclude');
    await page.locator('#review-rationale').fill('Synthetic review: inspect corresponding slices and guard bands.');
    await page.locator('#acknowledge-issues').check();
    await page.locator('#save-review').click();
    await page.waitForFunction(() => document.getElementById('review-save-state').textContent.includes('Review saved'));
    assert(saved.acknowledge_issues);
    assert.equal(saved.split_mode, 'spatial');
    assert.equal(await page.evaluate(() => Array.from(
      XrayUpload.sha256(new TextEncoder().encode('abc')), value => value.toString(16).padStart(2, '0'),
    ).join('')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');

    batch.status = 'cloud_ready'; batch.updated_at++;
    batch.version = {
      id: 'synthetic-version', manifest: {counts: {train: 2, monitor: 0, test: 0}},
      cloud: {uri: 'gs://synthetic-test-only/data.tar'},
    };
    await page.evaluate(() => refreshBatch());
    assert(await page.locator('#review-section').isVisible());
    for (const id of ['save-review', 'freeze', 'reject']) {
      assert(await page.locator('#' + id).isHidden());
      assert(await page.locator('#' + id).isDisabled());
    }
    assert(await page.locator('[data-field=split]').first().isDisabled());
    assert(await page.locator('[data-field=gt_id]').first().isDisabled());
    assert(await page.locator('#review-rationale').isDisabled());
    assert(await page.locator('#pair-filter').isEnabled());
    await page.locator('#pair-filter').selectOption('excluded');
    assert.equal(await page.locator('.pair-card').count(), 2);
    await page.locator('#pair-filter').selectOption('all');
    await page.evaluate(() => refreshBatch());
    assert(await page.locator('[data-field=split]').first().isDisabled());
    await page.screenshot({path: path.join(artifacts, 'desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path: path.join(artifacts, 'mobile.png'), fullPage: true});
    assert(conditionalResponses > 0);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({
      passed: true, artifacts,
      checks: [
        'uploader-review-separation', 'reviewer-signin-full-refresh', 'pair-previews', 'compatible-GT-options',
        'spatial-range-all-conditions', 'review-save-acknowledgement', 'HTTP-full-SHA256',
        'mobile-overflow', 'no-browser-errors', 'frozen-review-visible-and-locked',
        'frozen-filter-functional', 'rerender-lock-preserved', 'conditional-polling', 'local-assets-only',
      ],
    }));
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
