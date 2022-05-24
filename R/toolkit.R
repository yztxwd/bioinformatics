# create GRange object given info
toGrange <- function(chr, start, end, ...){
  require(GenomicRanges)
  do.call(GRanges,
          list(seqnames = chr,
               ranges = IRanges(start=start,
                                end=end),
               ...))
}

# peak annotation
