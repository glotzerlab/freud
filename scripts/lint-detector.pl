use strict;
use warnings;
use File::Find;
# use Text::Diff::Parser;
use File::Temp;

# parameters for controlling what gets reported
my $max_line_len = 120;
my $overlength_threshold = 25;

# process all the lines in the file and check for tabs, eof newline, overlength conditions, etc.
sub process_file_lines
    {
    my $fname = $_[0];
    my $fullpath = $_[1];
    # for checking the final newline
    my $last_line;

    # open the file
    open(FILE, "< $fname") or die "can't open $fname: $!";

    # initialize counters to 0
    my $tab_count = 0;
    my $eol_whitespace_count = 0;
    my $line_count = 0;
    my $overlength_count = 0;
    my $has_doxygen_file = 0;
    my $has_doxygen_package = 0;

    # loop through all lines in the file and add up counters
    while (<FILE>)
        {
        $tab_count += tr/\t//;
        $last_line = $_ if eof;
        chomp();
        $eol_whitespace_count += /(\s*)$/ && length($1);

        if (length($_) > $max_line_len)
            {
            $overlength_count += 1;
            }

        if ($_ =~ /\\file (.*)$/)
            {
            $has_doxygen_file = 1;
            }

        if ($_ =~ /\\package (.*)$/)
            {
            $has_doxygen_package = 1;
            }
        $line_count += 1;
        }
    close(FILE);

    my $message = "";

    if ($tab_count > 0)
        {
        $message .= "tabs:                $tab_count\n";
        }
    if ($eol_whitespace_count > 0)
        {
        $message .= "EOL whitespace:      $eol_whitespace_count\n";
        }
    if ($overlength_count > $overlength_threshold)
        {
        $message .= "lines overlength:    $overlength_count\n";
        }
    if (!$has_doxygen_file && !($fname =~ /\.py$/ or $fullpath =~ /\/unit_tests\//))
        {
        $message .= "missing doxygen \\file\n";
        }
    if (!$has_doxygen_package && ($fullpath =~ /\/pymodule\//))
        {
        $message .= "missing doxygen \\package\n";
        }


    return ($message, $line_count);
    }

sub wanted
    {
    my $fname = $_;

    # skip processing if this file is in the extern directory
    if ($File::Find::name =~ /\/extern\//)
        {
        return;
        }

    # skip processing if this file is in the build
    if ($File::Find::name =~ /\/build\//)
        {
        return;
        }

    # skip processing if this file is in the build
    if ($File::Find::name =~ /\/cpp\/molfile\//)
        {
        return;
        }

    # skip processing if this file is in the microbenchmarks directory
    if ($File::Find::name =~ /\/benchmarks\//)
        {
        return;
        }

    if (/\.cc$/ or /\.h$/ or /\.cu$/ or /\.cuh$/ or /\.py$/)
        {
        my $full_message = "";
        my $message;
        my $line_count;
        ($message, $line_count) = process_file_lines($fname, $File::Find::name);
        $full_message .= $message;
        #$full_message .= process_file_astyle($fname, $line_count);

        if ($full_message)
            {
            print "$File::Find::name\n";
            print $full_message;
            print "\n";
            }
        }
    }

# grep through the source and look for problems
finddepth(\&wanted, '.');
