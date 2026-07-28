# Pure-Ruby image dimension reader (PNG/JPEG/GIF) exposed as the
# `img_dimensions` Liquid filter. figure.liquid uses it to emit an
# aspect-ratio hint so the browser can reserve layout space before the
# image loads (avoids CLS) without changing the rendered size.
module Jekyll
  module ImageDimensions
    CACHE = {}

    # Returns [width, height] in pixels, or nil when unknown
    # (missing file, unsupported format, remote URL).
    def img_dimensions(path)
      return nil if path.nil?

      path = path.to_s
      return nil if path.empty? || path.start_with?('http', '//')
      return CACHE[path] if CACHE.key?(path)

      CACHE[path] = read_dimensions(path)
    end

    private

    def read_dimensions(path)
      return nil unless File.file?(path)

      File.open(path, 'rb') do |f|
        header = f.read(26)
        return nil if header.nil?

        if header.start_with?("\x89PNG".b)
          [header[16, 4].unpack1('N'), header[20, 4].unpack1('N')]
        elsif header.start_with?('GIF8'.b)
          [header[6, 2].unpack1('v'), header[8, 2].unpack1('v')]
        elsif header.start_with?("\xFF\xD8".b)
          jpeg_dimensions(f)
        end
      end
    rescue SystemCallError
      nil
    end

    def jpeg_dimensions(file)
      file.seek(2)
      loop do
        byte = file.read(1)
        return nil if byte.nil?
        next unless byte == "\xFF".b

        marker = file.read(1)
        return nil if marker.nil?
        next if marker == "\xFF".b

        m = marker.unpack1('C')
        next if m == 0xD8 || m == 0x01 || (0xD0..0xD7).cover?(m)

        len = file.read(2)&.unpack1('N')
        return nil if len.nil? || len < 2

        if (0xC0..0xCF).cover?(m) && ![0xC4, 0xC8, 0xCC].include?(m)
          data = file.read(5)
          return nil if data.nil? || data.bytesize < 5

          return [data[3, 2].unpack1('N'), data[1, 2].unpack1('N')]
        end

        file.seek(len - 2, IO::SEEK_CUR)
      end
    rescue SystemCallError
      nil
    end
  end
end

Liquid::Template.register_filter(Jekyll::ImageDimensions)
