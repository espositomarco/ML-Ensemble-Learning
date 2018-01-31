datasets = {
	"student" : {
		"train_name" : "prep_data/student/student_grades.csv_preprocessed",
		"X_col" : range(33),
		"Y_col" : [33],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"contraceptive" : {
		"train_name" : "prep_data/contraceptive/contraceptive.csv_preprocessed",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"autism" : {
		"train_name" : "prep_data/Autism-Adult-Data/Autism-Adult-Data-preproc.csv_preprocessed",
		"X_col" : range(20),
		"Y_col" : [20],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"bankruptcy" : {
		"train_name" : "prep_data/bankruptcy/bankrupt.csv_preprocessed",
		"X_col" : range(6),
		"Y_col" : [6],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"breast_cancer" : {
		"train_name" : "prep_data/breast-cancer/breast-cancer-wisconsin.data_preprocessed",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"horse" : {
		"train_name" : "prep_data/horse-colic/horse-colic.data-preproc.csv_preprocessed",
		"X_col" : range(22),
		"Y_col" : [22],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
    "hr" : {
		"train_name" : "prep_data/hr-analytics/HR_comma_sep.csv_preprocessed_reduced",
		"X_col" : range(10),
		"Y_col" : [10],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"english" : {
		"train_name" : "prep_data/teaching-english/tae.csv_preprocessed",
		"X_col" : range(5),
		"Y_col" : [5],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"phishing" : {
		"train_name" : "prep_data/website-phishing/PhishingData.csv_preprocessed",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"wine" : {
		"train_name" : "prep_data/wine-quality/winequality-white.csv_preprocessed",
		"X_col" : range(11),
		"Y_col" : [11],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	#"amazon" : {
	#	"train_name" : "prep_data/amazon/amazon_pca.csv",
	#	"X_col" : range(1,562),
	#	"Y_col" : [0],
	#	"has_header" : True,
	#	"filetype" : "CSV",
	#	"encode_labels" : False
	#},
	"congress" : {
		"train_name" : "prep_data/congress/congress_leave.csv",
		"X_col" : range(2,18),
		"Y_col" : [1],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : True
	},
	#"covertype" : {
	#	"train_name" : "prep_data/covertype/covertype_scale.csv",
	#	"X_col" : range(54),
	#	"Y_col" : [54],
	#	"has_header" : True,
	#	"filetype" : "CSV",
	#	"encode_labels" : True
	#}
}
