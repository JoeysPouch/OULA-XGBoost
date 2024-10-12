-- Defines course code table
CREATE TABLE courses (code_module VARCHAR(45),
	code_presentation VARCHAR(45),
    module_presentation_length INT,
	PRIMARY KEY (code_module, code_presentation));

-- Defines table of student demographics
CREATE TABLE studentInfo (code_module VARCHAR(45),
	code_presentation VARCHAR(45),
	id_student INT,
	gender VARCHAR(3),
	imd_band VARCHAR(16),
	highest_education VARCHAR(45),
	age_band VARCHAR(16),
	number_of_prev_attempts INT,
	studied_credits INT,
	region VARCHAR(45),
	disability VARCHAR(3),
	final_result VARCHAR(45),
	PRIMARY KEY (id_student, code_module, code_presentation),
	FOREIGN KEY (code_module, code_presentation) 
	REFERENCES courses(code_module, code_presentation));

-- Defines assessment code table
CREATE TABLE assessments (code_module VARCHAR(45),
	code_presentation VARCHAR(45),
	id_assessment INT PRIMARY KEY,
	assessment_type VARCHAR(45),
	assessment_date INT,
    weight FLOAT,
	FOREIGN KEY (code_module, code_presentation) 
	REFERENCES courses(code_module, code_presentation));

-- Defines Virtual Learning Environment code table
CREATE TABLE vle (id_site INT PRIMARY KEY,
	code_module VARCHAR(45),
	code_presentation VARCHAR(45),
	activity_type VARCHAR(45),
	week_from INT,
	week_to INT,
	FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation));

-- Defines table containing data of student interaction with the Virtual Learning Environment
CREATE TABLE studentVle (id_site INT,
	id_student INT,
	code_module VARCHAR(45),
	code_presentation VARCHAR(45),
	assessment_date INT,
	sum_click INT,
	FOREIGN KEY (id_student, code_module, code_presentation) REFERENCES studentInfo(id_student, code_module, code_presentation),
	FOREIGN KEY (id_site) REFERENCES vle(id_site),
	FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation));
				
-- Defines table containing student assessment performance data						 
CREATE TABLE studentAssessment (id_student INT,
	id_assessment INT,
	date_submitted INT,
	score FLOAT NULL,
	FOREIGN KEY (id_student, code_module, code_presentation) REFERENCES studentInfo(id_student, code_module, code_presentation),
	FOREIGN KEY (id_assessment) REFERENCES assessments(id_assessment));

-- Defines table containing data on individual students
CREATE TABLE studentRegistration (code_module VARCHAR(45),
	code_presentation VARCHAR(45),
	id_student INT,
	date_registration INT,
	date_unregistration INT,
	FOREIGN KEY (id_student, code_module, code_presentation) REFERENCES studentInfo(id_student, code_module, code_presentation),
	FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation));

-- Adds total_clicks column to StudentInfo
ALTER TABLE studentInfo
ADD COLUMN total_clicks INT DEFAULT 0;

-- Adds the total_clicks data to the studentInfo table
UPDATE studentInfo
SET total_clicks = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation

-- Adds the activity_type column to studentVle
UPDATE studentVle
SET activity_type = vle.activity_type
FROM vle
WHERE studentVle.id_site = vle.id_site;

-- Gets the number of clicks of each activity type
SELECT activity_type, COUNT(*) FROM studentVle
GROUP BY activity_type;

-- Adds columns for total_clicks by specific types of resource to StudentInfo
ALTER TABLE studentInfo
ADD COLUMN total_clicks_forumng INT DEFAULT 0,
ADD COLUMN total_clicks_homepage INT DEFAULT 0,
ADD COLUMN total_clicks_oucontent INT DEFAULT 0,
ADD COLUMN total_clicks_ouwiki INT DEFAULT 0,
ADD COLUMN total_clicks_quiz INT DEFAULT 0;

-- Adds the total_clicks_forumng data to the studentInfo table
UPDATE studentInfo
SET total_clicks_forumng = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
WHERE activity_type = 'forumng'
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation;

-- Adds the total_clicks_homepage data to the studentInfo table
UPDATE studentInfo
SET total_clicks_homepage = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
WHERE activity_type = 'homepage'
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation;

-- Adds the total_clicks_oucontent data to the studentInfo table
UPDATE studentInfo
SET total_clicks_oucontent = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
WHERE activity_type = 'oucontent'
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation;

-- Adds the total_clicks_ouwiki data to the studentInfo table
UPDATE studentInfo
SET total_clicks_ouwiki = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
WHERE activity_type = 'ouwiki'
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation;

-- Adds the total_clicks_quiz data to the studentInfo table
UPDATE studentInfo
SET total_clicks_quiz = subquery.total_clicks
FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click) AS total_clicks 
FROM 
	  studentVle
WHERE activity_type = 'quiz'
GROUP BY id_student, code_module, code_presentation) AS subquery
WHERE studentInfo.id_student = subquery.id_student
AND studentInfo.code_module = subquery.code_module
AND studentInfo.code_presentation = subquery.code_presentation;

-- Adds column for concatenated student id, module code and module presentation code
ALTER TABLE studentInfo
ADD COLUMN student_module_pres VARCHAR(25);

-- Concatenates student id, module code and module presentation code
UPDATE studentInfo
SET student_module_pres = id_student || '-' || code_module || '-' || code_presentation

-- Adds column for concatenated module code and module presentation code
ALTER TABLE studentInfo
ADD COLUMN module_pres VARCHAR(25);

-- Concatenates module code and module presentation code
UPDATE studentInfo
SET module_pres = code_module || '-' || code_presentation;

-- Loops through each type of learning resource and finds the sum of clicks for each student on that resource
DO $$
DECLARE
	activity RECORD;
	col_name VARCHAR(45);
BEGIN
	FOR activity in 
		SELECT DISTINCT activity_type
		FROM vle
	LOOP
		col_name := 'total_clicks_' || activity.activity_type;
	
		EXECUTE format('ALTER TABLE studentInfo
					   ADD COLUMN %I INT DEFAULT 0;', col_name);
		
		EXECUTE format('UPDATE studentInfo
					   SET %I = subquery.total_clicks
					   	FROM (SELECT id_student, code_module, code_presentation, SUM(sum_click)
						AS total_clicks
					   	FROM studentVle
					   WHERE activity_type = %L
					   GROUP BY id_student, code_module, code_presentation) AS subquery
					   WHERE studentInfo.id_student = subquery.id_student
              		   AND studentInfo.code_module = subquery.code_module
             		   AND studentInfo.code_presentation = subquery.code_presentation;',
					   col_name, activity.activity_type
					   );
		END LOOP;

END $$;


-- Joins together modules where 
UPDATE studentInfo
SET pres_groups =
  CASE
    WHEN code_module = 'AAA' AND code_presentation = '2013J' THEN 0
    WHEN code_module = 'AAA' AND code_presentation = '2014J' THEN 0
    WHEN code_module = 'BBB' AND code_presentation = '2013B' THEN 1
	WHEN code_module = 'BBB' AND code_presentation = '2014B' THEN 1
	WHEN code_module = 'BBB' AND code_presentation = '2014J' THEN 2
    WHEN code_module = 'BBB' AND code_presentation = '2013J' THEN 2
    WHEN code_module = 'CCC' AND code_presentation = '2014B' THEN 3
    WHEN code_module = 'CCC' AND code_presentation = '2014J' THEN 4
    WHEN code_module = 'DDD' AND code_presentation = '2013B' THEN 5
	WHEN code_module = 'DDD' AND code_presentation = '2014B' THEN 5
    WHEN code_module = 'DDD' AND code_presentation = '2013J' THEN 6
    WHEN code_module = 'DDD' AND code_presentation = '2014J' THEN 6
    WHEN code_module = 'EEE' AND code_presentation = '2013J' THEN 7
	WHEN code_module = 'EEE' AND code_presentation = '2014J' THEN 7
    WHEN code_module = 'EEE' AND code_presentation = '2014B' THEN 8
    WHEN code_module = 'FFF' AND code_presentation = '2013B' THEN 9
    WHEN code_module = 'FFF' AND code_presentation = '2014B' THEN 9
	WHEN code_module = 'FFF' AND code_presentation = '2013J' THEN 10
    WHEN code_module = 'FFF' AND code_presentation = '2014J' THEN 10
    WHEN code_module = 'GGG' AND code_presentation = '2013J' THEN 11
	WHEN code_module = 'GGG' AND code_presentation = '2014J' THEN 11
    WHEN code_module = 'GGG' AND code_presentation = '2014B' THEN 12
    ELSE NULL
  END

-- Adds pres_groups to studentVle

ALTER TABLE studentVle
ADD COLUMN pres_groups INT;

UPDATE studentVle
SET pres_groups =
  CASE
    WHEN code_module = 'AAA' AND code_presentation = '2013J' THEN 0
    WHEN code_module = 'AAA' AND code_presentation = '2014J' THEN 0
    WHEN code_module = 'BBB' AND code_presentation = '2013B' THEN 1
	WHEN code_module = 'BBB' AND code_presentation = '2014B' THEN 1
	WHEN code_module = 'BBB' AND code_presentation = '2014J' THEN 2
    WHEN code_module = 'BBB' AND code_presentation = '2013J' THEN 2
    WHEN code_module = 'CCC' AND code_presentation = '2014B' THEN 3
    WHEN code_module = 'CCC' AND code_presentation = '2014J' THEN 4
    WHEN code_module = 'DDD' AND code_presentation = '2013B' THEN 5
	WHEN code_module = 'DDD' AND code_presentation = '2014B' THEN 5
    WHEN code_module = 'DDD' AND code_presentation = '2013J' THEN 6
    WHEN code_module = 'DDD' AND code_presentation = '2014J' THEN 6
    WHEN code_module = 'EEE' AND code_presentation = '2013J' THEN 7
	WHEN code_module = 'EEE' AND code_presentation = '2014J' THEN 7
    WHEN code_module = 'EEE' AND code_presentation = '2014B' THEN 8
    WHEN code_module = 'FFF' AND code_presentation = '2013B' THEN 9
    WHEN code_module = 'FFF' AND code_presentation = '2014B' THEN 9
	WHEN code_module = 'FFF' AND code_presentation = '2013J' THEN 10
    WHEN code_module = 'FFF' AND code_presentation = '2014J' THEN 10
    WHEN code_module = 'GGG' AND code_presentation = '2013J' THEN 11
	WHEN code_module = 'GGG' AND code_presentation = '2014J' THEN 11
    WHEN code_module = 'GGG' AND code_presentation = '2014B' THEN 12
    ELSE NULL
  END



