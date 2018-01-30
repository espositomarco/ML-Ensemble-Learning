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
	}

}