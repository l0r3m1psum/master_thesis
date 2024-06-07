# https://texdoc.org/serve/latexmk/0
# https://github.com/e-dschungel/latexmk-config/blob/master/latexmkrc#L112
print "LKJDNALKJBNFAKSJNADSKLJFNSLKNASASFKASFLKSAFAFSDLKBJ";
system "mpost figures.mp";
add_cus_dep("gp", "ps", 0, 'mysubroutine');
sub mysubroutine {
    print "LKJDNALKJBNFAKSJNADSKLJFNSLKNASASFKASFLKSAFAFSDLKBJ";
    return 0;
}