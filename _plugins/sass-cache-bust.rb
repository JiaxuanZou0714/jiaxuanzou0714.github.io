# based on https://github.com/george-gca/jekyll-cache-bust
#
# The upstream `bust_css_cache` filter hardcodes `assets/_sass` as the Sass
# directory, which does not exist in this project (Sass sources live in
# `_sass/`). The digest therefore collapses to the MD5 of an empty string
# (d41d8cd98f00b204e9800998ecf8427e) and never changes, so browsers keep
# serving stale main.css after style updates.
#
# The `bust_sass_cache` filter below hashes the real Sass sources: the
# entry file (assets/css/main.scss) plus everything under `_sass/`.
module Jekyll
  module SassCacheBust
    require 'digest/md5'

    SASS_ENTRY_FILE = 'assets/css/main.scss'
    SASS_SOURCE_DIR = '_sass'

    def bust_sass_cache(file_name)
      sources = Dir[File.join(SASS_SOURCE_DIR, '**', '*')].reject { |f| File.directory?(f) }
      sources << SASS_ENTRY_FILE
      content = sources.sort.map { |f| File.binread(f) }.join
      [file_name, '?v=', Digest::MD5.hexdigest(content)].join
    end
  end
end

Liquid::Template.register_filter(Jekyll::SassCacheBust)
