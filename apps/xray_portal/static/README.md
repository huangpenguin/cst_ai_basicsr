# Portal browser assets

The page is served from work1; no CDN is contacted. `vendor/upload.js` bundles
`tus-js-client` 4.3.1 and `@noble/hashes` 2.4.0. Rebuild with Node and npm:

```sh
cd apps/xray_portal/static
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

The lockfile and generated bundle are committed; `node_modules` is not.
Third-party licenses are in `vendor/`.

The browser hashes the entire selected file in 8 MiB slices before registering
an upload. This uses incremental SHA256 JavaScript, so it works on the internal
HTTP pilot without Web Crypto's secure-context requirement. Uploads use 32 MiB
tus chunks, at most two concurrent files, and a fingerprint containing the
batch, file ID and complete content hash. After a page reload, the user must
select the original local file again. The service independently checks the
received hash; a completed transfer is not yet validated TIFF data.

Review controls keep mapping approval separate from split assignment. Group
controls assign all conditions together; spatial controls assign an inclusive
slice range across all conditions in one group. The server remains authoritative
for authorization, valid mappings, split overlap, guard bands, file checks,
freezing and GCS publication. UI hiding alone is not an access-control boundary.

Frozen versions retain a read-only review panel: the approved GT mapping, split,
notes, previews and original-image download links remain visible. Filtering and
pagination work, while every review mutation control is disabled. Unchanged
batch polls receive a small conditional response; batch selection and reviewer
sign-in explicitly request the full report.

## Browser regression tests

The checked-in test uses synthetic API responses and geometric preview images;
it never connects to the live portal, GCS or company image files. Playwright is
pinned to 1.58.2. Install the browser separately (npm installation does not
implicitly download it):

```sh
cd apps/xray_portal/static
npm ci --ignore-scripts --no-audit --no-fund
npx playwright install chromium
npm run test:browser
```

For an existing compatible Chrome installation, skip the browser download and
run `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/google-chrome npm run test:browser`.
The test resolves repository files relative to its own location and writes
synthetic screenshots to a new system temporary directory, whose path is printed
on success. It checks role separation, compatible GT mapping choices, spatial
range assignment, frozen review locks, conditional polling, HTTP-compatible
hashing, local-only assets and mobile layout.
