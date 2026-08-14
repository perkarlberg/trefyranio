import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// Static site (default output). Served free on Firebase Hosting.
export default defineConfig({
  site: 'https://trefyran.io',
  // Firebase Hosting canonicalizes to no-trailing-slash (cleanUrls +
  // trailingSlash:false in web/firebase.json) and 301s "/metod/" -> "/metod".
  // Astro's default emits the slashed form, so every sitemap entry and every
  // canonical pointed at a URL that redirects — 15 of 16 sitemap URLs were 301s,
  // burning crawl budget on a domain that has very little. Match Firebase.
  trailingSlash: 'never',
  integrations: [
    sitemap({
      // lastmod = build time: the whole site is regenerated each forecast deploy
      // (every page embeds forecast data / carries the "Uppdaterad" footer), so
      // this is an honest freshness signal — the one sitemap field Google actually
      // uses to schedule re-crawls. (changefreq/priority are largely ignored.)
      lastmod: new Date(),
      changefreq: 'daily',
      priority: 0.7,
      serialize(item) {
        // List the URLs Firebase actually serves 200 on rather than redirects.
        // The integration enforces the config's trailingSlash:'never' after this
        // hook, so the root ends up as the bare origin — equivalent to "/" per
        // RFC 3986 and normalized by crawlers, so we leave it alone.
        item.url = item.url.replace(/(.)\/$/, '$1');
        if (item.url.replace(/\/$/, '') === 'https://trefyran.io') item.priority = 1.0;
        else if (item.url.includes('/parti/')) item.priority = 0.6;
        return item;
      },
    }),
  ],
});
