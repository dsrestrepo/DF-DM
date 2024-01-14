-- Codes: http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf
-- http://efele.net/maps/fips-10/data/fips-all.txt
SELECT
  GLOBALEVENTID, SQLDATE, MonthYear, Year, Actor1Name, Actor1CountryCode, Actor2Name, Actor2CountryCode, EventCode, 
  NumMentions, NumSources, NumArticles, AvgTone, ActionGeo_CountryCode, ActionGeo_ADM1Code, ActionGeo_Lat, ActionGeo_Long, 
  ActionGeo_FeatureID, SOURCEURL
FROM
  `gdelt-bq.full.events`
WHERE
  (ActionGeo_CountryCode = 'CO' AND ActionGeo_ADM1Code = 'CO20') AND
  SQLDATE >= 20150101 AND
  (EventCode LIKE '18%' OR EventCode LIKE '19%' OR EventCode LIKE '20%') -- AND
  -- (LOWER(SOURCEURL) LIKE '%violence%' OR LOWER(SOURCEURL) LIKE '%violencia de genero%' OR 
  --LOWER(SOURCEURL) LIKE '%violencia domestica%' OR LOWER(SOURCEURL) LIKE '%violencia contra la mujer%' OR
  -- LOWER(SOURCEURL) LIKE '%abuso%' OR LOWER(SOURCEURL) LIKE '%violacion%')