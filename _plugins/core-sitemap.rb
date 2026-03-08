module Jekyll
  class CoreSitemapGenerator < Generator
    safe true
    priority :lowest

    def generate(site)
      patterns = Array(site.config['core_sitemap_include_patterns']).map { |pattern| Regexp.new(pattern) }
      return if patterns.empty?

      documents(site).each do |document|
        next unless document.respond_to?(:url)

        url = document.url.to_s
        next if url.empty?
        next if included_in_core_sitemap?(url, patterns)

        document.data ||= {}
        document.data['sitemap'] = false
      end
    end

    private

    def documents(site)
      site.pages + site.collections.values.flat_map(&:docs)
    end

    def included_in_core_sitemap?(url, patterns)
      patterns.any? { |pattern| pattern.match?(url) }
    end
  end
end