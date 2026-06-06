import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// Static site (default output). Served free on Firebase Hosting.
export default defineConfig({
  site: 'https://trefyran.io',
  integrations: [sitemap()],
});
