datasets = {
	"student" : {
		"train_name" : "prep_data/student/student_grades.csv",
		"X_col" : range(33),
		"Y_col" : [33],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : True
	},
	"bank" : {
		"train_name" : "prep_data/bankruptcy/bankrupt.arff",
		"X_col" : range(6),
		"Y_col" : [6],
		"has_header" : False,
		"filetype" : "arff",
		"encode_labels" : True
	},
	"contraceptive" : {
		"train_name" : "prep_data/contraceptive/contraceptive.csv",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : False,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"amazon" : {
		"train_name" : "prep_data/amazon/amazon_pca.csv",
		"X_col" : range(1,562),
		"Y_col" : [0],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"congress" : {
		"train_name" : "prep_data/congress/congress_leave.csv",
		"X_col" : range(2,18),
		"Y_col" : [1],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : True
	},
	"covertype" : {
		"train_name" : "prep_data/covertype/covertype_scale.csv",
		"X_col" : range(54),
		"Y_col" : [54],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : True
	}

}