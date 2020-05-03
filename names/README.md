# Name classification

Some experiments with trying to predict if a given name is a company 
or person name. This could be used as a predictive layer for both
structured data and as a second-stage filter for a named entity
extractor.


## Training data prep

Filtering the name prop dump:

```bash
grep -v "\"\(schema\|PublicBody\|LegalEntity\)" names.csv >names.filtered.csv
```

Shuffling the names:

```bash
../../terashuf/terashuf < names.filtered.csv >names.filtered.shuffled.csv
```

* `terashuf`, shuffle very large files: https://github.com/alexandres/terashuf

Class imbalance:

```bash
grep "\"Company" names.filtered.csv | wc -l 
grep "\"Person" names.filtered.csv | wc -l
```



## Credits

Built on by previous work by: 

* Rinat Tuhvatshin (Kloop.kg)
* Alexey Guliayev (Kloop.kg)
* Jeremy Merill (Quartz AI)