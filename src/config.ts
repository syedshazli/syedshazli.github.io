type Site = {
  website: string;
  author: string;
  profile: string;
  desc: string;
  title: string;
  ogImage: string;
  lightAndDarkMode: boolean;
  postPerIndex: number;
  postPerPage: number;
  scheduledPostMargin: number;
  showArchives: boolean;
  showBackButton: boolean;
  editPost: {
    url: string;
    text: string;
    appendFilePath: boolean;
    enabled: boolean;
  };
  dynamicOgImage: boolean;
  dir: string;
  lang: string;
  timezone: string;
};
export const SITE: Site = {
  website: "https://syedshazli.github.io/syedshazlii/", // Change this!
  author: "Syed Shazli", // Your name
  profile: "https://syedshazli.github.io/", // Your profile URL
  desc: "Your site description here", // Update this
  title: "Your Site Title", // Update this
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000,
  showArchives: true,
  showBackButton: true,
  editPost: {
    url: "https://github.com/syedshazli/syedshazlii/edit/main/src/content/blog",
    text: "Suggest Changes",
    appendFilePath: true,
    enabled: true,
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
