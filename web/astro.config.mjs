import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// Static site (default output). Served free on Firebase Hosting.
export default defineConfig({
  site: 'https://trefyran.io',
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
        if (item.url === 'https://trefyran.io/') item.priority = 1.0;
        else if (item.url.includes('/parti/')) item.priority = 0.6;
        return item;
      },
    }),
  ],
});
