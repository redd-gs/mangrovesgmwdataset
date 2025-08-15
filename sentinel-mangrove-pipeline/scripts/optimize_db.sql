-- Optimize database performance by creating indexes on frequently queried columns
CREATE INDEX idx_gid ON public.gmw_2016_v2 (gid);
CREATE INDEX idx_geom ON public.gmw_2016_v2 USING GIST (geom);

-- Analyze the table to update statistics for the query planner
ANALYZE public.gmw_2016_v2;

-- Vacuum the table to reclaim storage and optimize performance
VACUUM (VERBOSE, ANALYZE) public.gmw_2016_v2;