# Work around WEBrick's RFC 9110 bug where matching If-Range + Range triggers
# 304 Not Modified instead of 206 Partial Content, breaking HTML5 <video>/<audio>
# playback and seeking in Chromium during local `jekyll serve`.
require "webrick"
require "webrick/httpservlet/filehandler"

module WEBrickIfRangeFix
  def not_modified?(req, res, mtime, etag)
    return false if req["range"]

    super
  end

  def do_GET(req, res)
    res["accept-ranges"] = "bytes"
    super
  end
end

WEBrick::HTTPServlet::DefaultFileHandler.prepend(WEBrickIfRangeFix)
